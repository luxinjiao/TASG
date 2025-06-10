# 使用 Miniconda 作为基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . .


