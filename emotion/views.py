from django.shortcuts import render

from emotion.models import TrainingStatus
from emotion.utils.inference import predict
import json
import time
import threading
from collections import Counter
from datetime import datetime

import torch
import evaluate
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

# 全局训练状态
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_step': 0,
    'total_steps': 0,
    'train_loss': 0.0,
    'eval_loss': 0.0,
    'eval_f1': 0.0,
    'eval_accuracy': 0.0,
    'eval_recall': 0.0,
    'progress_percent': 0.0,
    'logs': [],
    'error_message': '',
    'final_metrics': {},
    'hyperparameters': {},
    'training_start_time': None,
    'training_end_time': None,
    'should_stop': False  # 停止标志
}

# 全局变量存储训练器和模型
current_trainer = None
tokenizer = None


def count_labels(ds):
    """统计标签分布"""
    label_counts = Counter(ds['label'])
    return label_counts.get(0, 0), label_counts.get(1, 0)


def preprocess_function(examples):
    """数据预处理函数"""
    global tokenizer
    return tokenizer(examples['text'], padding=True, truncation=True)


def eval_metric(eval_predict):
    """评估指标计算"""
    acc_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)

    acc = acc_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    result = {}
    result.update(acc)
    result.update(rec)
    result.update(f1)
    return result


class TrainingProgressCallback(TrainerCallback):
    """训练进度回调类"""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时的回调"""
        global training_status
        training_status.update({
            'is_training': True,
            'training_start_time': datetime.now().isoformat(),
            'total_epochs': args.num_train_epochs,
            'total_steps': state.max_steps,
            'should_stop': False
        })
        training_status['logs'].append(f"🚀 开始训练，总共 {args.num_train_epochs} 轮，{state.max_steps} 步")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的回调"""
        global training_status
        training_status.update({
            'training_end_time': datetime.now().isoformat(),
            'progress_percent': 100.0
        })
        training_status['logs'].append("🏁 训练完成！")

    def on_epoch_begin(self, args, state, control, **kwargs):
        """每轮开始时的回调"""
        global training_status
        training_status['current_epoch'] = int(state.epoch) + 1
        training_status['logs'].append(f"📚 开始第 {training_status['current_epoch']} 轮训练")

    def on_step_end(self, args, state, control, **kwargs):
        """每步结束时的回调"""
        global training_status, current_trainer

        # 检查是否需要停止训练
        if training_status['should_stop']:
            training_status['logs'].append("⏹️ 收到停止信号，正在安全停止训练...")
            control.should_training_stop = True
            return

        self.step_count += 1
        training_status['current_step'] = self.step_count

        # 更新进度百分比
        if training_status['total_steps'] > 0:
            progress = (self.step_count / training_status['total_steps']) * 100
            training_status['progress_percent'] = min(progress, 100.0)

        # 获取最新的训练损失
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                training_status['train_loss'] = last_log['loss']

        # 每10步记录一次日志
        if self.step_count % 10 == 0:
            training_status['logs'].append(
                f"📊 步骤 {self.step_count}/{training_status['total_steps']}, "
                f"损失: {training_status['train_loss']:.4f}"
            )

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """评估时的回调"""
        global training_status
        if logs:
            # 更新评估指标
            training_status.update({
                'eval_loss': logs.get('eval_loss', 0.0),
                'eval_accuracy': logs.get('eval_accuracy', 0.0),
                'eval_recall': logs.get('eval_recall', 0.0),
                'eval_f1': logs.get('eval_f1', 0.0)
            })

            # 添加评估日志
            training_status['logs'].append(
                f"📈 评估结果 - 准确率: {logs.get('eval_accuracy', 0):.4f}, "
                f"F1: {logs.get('eval_f1', 0):.4f}, "
                f"召回率: {logs.get('eval_recall', 0):.4f}"
            )

@csrf_exempt
def train_model_async():
    """异步训练模型函数"""
    global training_status, current_trainer, tokenizer

    try:
        training_status['logs'].append("📦 正在加载数据集...")

        # 加载数据集
        dataset = load_dataset("lansinuote/ChnSentiCorp", cache_dir="data")
        train_dataset = dataset['train']
        valid_dataset = dataset['validation']
        test_dataset = dataset['test']

        training_status['logs'].append(
            f"✅ 数据集加载完成 - 训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")

        # 统计标签分布
        train_neg, train_pos = count_labels(train_dataset)
        valid_neg, valid_pos = count_labels(valid_dataset)
        training_status['logs'].append(f"📊 训练集标签分布 - 负样本: {train_neg}, 正样本: {train_pos}")
        training_status['logs'].append(f"📊 验证集标签分布 - 负样本: {valid_neg}, 正样本: {valid_pos}")

        # 加载模型和tokenizer
        model_name = "bert-base-chinese"
        training_status['logs'].append(f"🤖 正在加载模型: {model_name}")

        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

        training_status['logs'].append("✅ 模型加载完成")

        # 数据预处理
        training_status['logs'].append("🔄 正在进行数据预处理...")
        encoded_datasets = dataset.map(preprocess_function, batched=True)
        training_status['logs'].append("✅ 数据预处理完成")

        # 训练参数
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            learning_rate=3e-5,
            metric_for_best_model="f1",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
        )

        # 记录超参数
        hyper_params = {
            "模型": model.__class__.__name__,
            "隐藏层大小": getattr(model.config, "hidden_size", "N/A"),
            "训练 epoch": training_args.num_train_epochs,
            "训练 batch_size": training_args.per_device_train_batch_size,
            "验证 batch_size": training_args.per_device_eval_batch_size,
            "学习率": training_args.learning_rate,
            "学习率 warm‑up 步数": training_args.warmup_steps,
            "权重衰减": training_args.weight_decay,
            "优化器": str(training_args.optim),
            "设备": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        training_status['hyperparameters'] = hyper_params

        training_status['logs'].append("⚙️ 训练参数配置完成")
        for key, value in hyper_params.items():
            training_status['logs'].append(f"   {key}: {value}")

        # 创建训练器
        current_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_datasets['train'],
            eval_dataset=encoded_datasets['validation'],
            compute_metrics=eval_metric,
            callbacks=[TrainingProgressCallback()],
        )

        training_status['logs'].append("🏃‍♂️ 开始模型训练...")

        # 开始训练
        current_trainer.train()

        # 如果训练被手动停止
        if training_status['should_stop']:
            training_status['logs'].append("⏹️ 训练已手动停止")
            return

        training_status['logs'].append("📊 正在评估训练集性能...")

        # 评估训练集
        train_metrics = current_trainer.evaluate(encoded_datasets["train"])
        train_result = {
            'accuracy': train_metrics['eval_accuracy'],
            'recall': train_metrics['eval_recall'],
            'f1': train_metrics['eval_f1']
        }

        training_status['logs'].append(
            f"📈 训练集结果 - 准确率: {train_result['accuracy']:.4f}, "
            f"召回率: {train_result['recall']:.4f}, F1: {train_result['f1']:.4f}"
        )

        training_status['logs'].append("📊 正在评估测试集性能...")

        # 评估测试集
        test_metrics = current_trainer.evaluate(encoded_datasets["test"])
        test_result = {
            'accuracy': test_metrics['eval_accuracy'],
            'recall': test_metrics['eval_recall'],
            'f1': test_metrics['eval_f1']
        }

        training_status['logs'].append(
            f"📈 测试集结果 - 准确率: {test_result['accuracy']:.4f}, "
            f"召回率: {test_result['recall']:.4f}, F1: {test_result['f1']:.4f}"
        )

        # 保存最终结果
        training_status['final_metrics'] = {
            'train': train_result,
            'test': test_result
        }

        # 保存模型
        training_status['logs'].append("💾 正在保存模型...")
        current_trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        training_status['logs'].append("✅ 模型保存完成")

        training_status['logs'].append("🎉 训练流程全部完成！")

        # 保存训练结果
        TrainingStatus.objects.all().delete()
        TrainingStatus.objects.create(**training_status)

    except Exception as e:
        error_msg = f"训练过程中发生错误: {str(e)}"
        training_status['error_message'] = error_msg
        training_status['logs'].append(f"❌ {error_msg}")
        training_status['is_training'] = False
    finally:
        training_status['is_training'] = False
        training_status['progress_percent'] = 100.0


@csrf_exempt
def start_training(request):
    """开始训练的API端点"""
    global training_status

    if training_status['is_training']:
        return JsonResponse({
            'success': False,
            'message': '模型正在训练中，请等待当前训练完成'
        })

    # 重置状态
    training_status.update({
        'is_training': False,  # 会在回调中设置为True
        'current_epoch': 0,
        'total_epochs': 0,
        'current_step': 0,
        'total_steps': 0,
        'train_loss': 0.0,
        'eval_loss': 0.0,
        'eval_f1': 0.0,
        'eval_accuracy': 0.0,
        'eval_recall': 0.0,
        'progress_percent': 0.0,
        'logs': [],
        'error_message': '',
        'final_metrics': {},
        'hyperparameters': {},
        'training_start_time': None,
        'training_end_time': None,
        'should_stop': False
    })

    # 在新线程中启动训练
    training_thread = threading.Thread(target=train_model_async)
    training_thread.daemon = True
    training_thread.start()

    return JsonResponse({
        'success': True,
        'message': '训练已开始'
    })

@csrf_exempt
def get_training_status(request):
    """获取训练状态的API端点"""
    global training_status
    return JsonResponse(training_status)

@csrf_exempt
def training_stream(request):
    """服务器发送事件(SSE)流式传输训练状态"""

    def json_default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def event_stream():
        global training_status
        last_step = -1
        last_log_count = 0

        while True:
            # 只有当状态改变时才发送数据
            current_step = training_status['current_step']
            current_log_count = len(training_status['logs'])

            if (current_step != last_step or
                    current_log_count != last_log_count or
                    not training_status['is_training']):
                yield f"data: {json.dumps(training_status, ensure_ascii=False, default=json_default)}\n\n"
                last_step = current_step
                last_log_count = current_log_count

            time.sleep(0.5)  # 每0.5秒检查一次

            # 如果训练完成，发送几次后停止
            if (not training_status['is_training'] and
                    training_status['progress_percent'] >= 100):
                for _ in range(5):  # 再发送5次确保前端收到
                    yield f"data: {json.dumps(training_status, ensure_ascii=False)}\n\n"
                    time.sleep(0.5)
                break

    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Access-Control-Allow-Origin'] = '*'
    return response


@csrf_exempt
def stop_training(request):
    """停止训练的API端点"""
    global training_status

    if training_status['is_training']:
        training_status['should_stop'] = True
        training_status['logs'].append("⏹️ 收到停止训练请求，正在安全停止...")
        return JsonResponse({
            'success': True,
            'message': '停止训练请求已发送，当前批次完成后将停止训练'
        })
    else:
        return JsonResponse({
            'success': False,
            'message': '当前没有正在进行的训练'
        })

@csrf_exempt
def get_training_results(request):
    """获取训练结果的API端点"""
    global training_status

    if not training_status['final_metrics']:
        if not TrainingStatus.objects.exists():
            return JsonResponse({
                'success': False,
                'message': '训练尚未完成或没有可用结果'
            })
        else:
            training_status = TrainingStatus.objects.values().first()
            return JsonResponse({
                'success': True,
                'data': {
                    'final_metrics': training_status.final_metrics,
                    'hyperparameters': training_status.hyperparameters,
                    'training_start_time': training_status.training_start_time,
                    'training_end_time': training_status.training_end_time,
                    'total_steps': training_status.total_steps,
                    'total_epochs': training_status.total_epochs
                }
            })

    return JsonResponse({
        'success': True,
        'data': {
            'final_metrics': training_status['final_metrics'],
            'hyperparameters': training_status['hyperparameters'],
            'training_start_time': training_status['training_start_time'],
            'training_end_time': training_status['training_end_time'],
            'total_steps': training_status['total_steps'],
            'total_epochs': training_status['total_epochs']
        }
    })

# Create your views here.
@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def inference(request):
    if request.method == 'GET':
        return render(request, 'blogs-2.html')

    data = json.loads(request.body)
    sentence = data.get('sentence', '')
    label, prob = predict(sentence)
    return JsonResponse({'label': label, 'prob': prob})

@csrf_exempt
def training(request):
    return render(request, 'service-page-2.html')


@csrf_exempt
def testing(request):
    return render(request, 'test.html')