# 配置 MySQL
1. 下载 MySQL
2. 在 emotionweb/settings.py 中修改相关用户配置

```plain
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': '',
        'USER': '',
        'PASSWORD': '',
        'HOST': '127.0.0.1',
        'PORT': 3306,
    }
}
```

# 配置 Python 环境
```plain
pip install -r requirements.txt
```

# 运行
1. 注意加上 --noreload 参数
2. 必须先训练，才能进行推理

