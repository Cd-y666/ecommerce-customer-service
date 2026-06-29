# 电商智能客服系统（RAG增强版）

基于 **Streamlit + LangChain + Ollama + ChromaDB** 的本地化电商智能客服系统。

**🔒 安全特性**：已部署文件上传防护、Prompt Injection 过滤、敏感信息过滤、速率限制等安全措施，可安全上传至 GitHub 或部署到公网。

---

## 📋 项目介绍

本项目是一个面向电商场景的智能客服系统，基于 RAG（检索增强生成）技术，能够回答商品咨询、推荐合适商品、解答售后政策等问题。

系统内置示例商品数据（**10款商品 + 14条FAQ**），也可上传自定义商品数据。完全本地化部署，无需外部 API，保护用户数据隐私。

**适用场景：**
- 电商平台智能客服
- 商品咨询自动化
- 售后政策解答
- 商品推荐系统

---

## ✨ 核心功能

- ✅ **商品知识库管理** — 支持上传商品数据 CSV、FAQ CSV、商品手册 PDF/DOCX/TXT
- ✅ **智能问答** — 回答商品规格、价格、库存、特点等问题
- ✅ **商品推荐** — 根据用户需求语义匹配并推荐合适商品（2-3个选项）
- ✅ **售后政策解答** — 解答退换货规则、保修政策等问题
- ✅ **多轮对话记忆** — 支持上下文连续对话，提升用户体验
- ✅ **来源追溯** — 回答标注信息来源，增强可信度
- ✅ **知识库持久化** — 基于 ChromaDB 的持久化存储，关闭重启不丢失
- ✅ **内置示例数据** — 10款商品 + 14条FAQ，开箱即用

---

## 🏗️ 技术架构

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit（Web 交互界面） |
| Agent 框架 | LangChain（ReAct Agent + 工具调用） |
| 本地 LLM | Ollama（deepseek-r1:7b） |
| Embedding 模型 | Ollama（shaw/dmeta-embedding-zh） |
| 向量数据库 | ChromaDB（本地持久化） |
| 数据格式 | CSV（商品数据 + FAQ）、PDF、DOCX、TXT |

---

## 🚀 快速开始

### 1. 安装 Ollama

下载并安装 Ollama：https://ollama.com

### 2. 下载模型

```bash
ollama pull deepseek-r1:7b
ollama pull shaw/dmeta-embedding-zh
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量（可选）

```bash
cp .env.example .env
# 根据需要编辑 .env 文件
```

### 5. 启动项目

```bash
streamlit run ecommerce_rag_bot.py
```

---

## 💡 使用方法

1. 左侧边栏勾选「使用示例数据」（默认已勾选）
2. 点击「构建知识库」按钮，等待完成
3. 在主界面输入问题，或点击快捷问题按钮
4. 系统会从商品知识库中检索并回答
5. 可上传自定义商品数据 CSV 文件，替换示例数据

---

## 📦 示例数据说明

系统内置 **10 款示例商品**，涵盖：
- 数码音频：智能蓝牙耳机、便携蓝牙音箱
- 电脑办公：轻薄商务笔记本
- 厨房家电：全自动咖啡机、空气炸锅
- 食品饮料：有机绿茶礼盒
- 运动户外：男士运动跑鞋
- 美妆护肤：补水面膜套装
- 玩具教育：儿童编程机器人
- 日用品：保温杯

内置 **14 条示例 FAQ**，涵盖产品咨询、使用指南、售后政策、推荐咨询等。

---

## 📊 自定义数据格式

### 商品数据 CSV 格式

必需字段：
```
商品ID, 商品名称, 分类, 品牌, 价格, 库存, 规格, 特点, 适用场景, 售后政策
```

示例：
```csv
P001,智能无线蓝牙耳机Pro,数码音频,声阔,299,156,蓝牙5.3/续航40h,入门级价格旗舰级降噪,通勤/运动,7天无理由退换
```

### FAQ CSV 格式

必需字段：
```
问题, 答案, 相关商品, 分类
```

示例：
```csv
"这款蓝牙耳机续航多久？","声阔智能无线蓝牙耳机Pro续航40小时","P001","产品咨询"
```

---

## 📁 项目结构

```
ecommerce-customer-service/
├── ecommerce_rag_bot.py  # 主程序
├── requirements.txt        # Python 依赖
├── .env.example          # 环境变量模板
├── .gitignore           # Git 忽略规则
├── README.md            # 本文件
└── ecommerce_chroma_db/ # （自动生成）知识库持久化目录
```

---

## ⚙️ 环境变量配置

复制 `.env.example` 为 `.env`，并根据需要修改：

```bash
cp .env.example .env
```

主要配置项：
- `OLLAMA_BASE_URL` — Ollama 服务地址（默认 `http://127.0.0.1:11434`）
- `LLM_MODEL` — LLM 模型名称（默认 `deepseek-r1:7b`）
- `EMBED_MODEL` — Embedding 模型名称（默认 `shaw/dmeta-embedding-zh`）
- `ECOM_CHROMA_DIR` — ChromaDB 持久化目录（默认 `./ecommerce_chroma_db`）
- `SUPPORT_HOTLINE` — 客服热线（在回答中展示）
- `SUPPORT_EMAIL` — 客服邮箱
- `SUPPORT_HOURS` — 客服服务时间

---

## 🔒 安全特性

本项目已部署以下安全措施：

| 安全类别 | 具体措施 |
|---------|---------|
| **文件安全** | 扩展名白名单、文件大小限制、路径遍历防护、文件名安全清洗 |
| **输入安全** | 长度限制、Prompt Injection 基础过滤 |
| **输出安全** | 思考标签清理、敏感信息过滤（手机号/身份证/路径） |
| **资源保护** | Agent 最大迭代次数限制、检索结果数量限制 |
| **速率限制** | 会话级消息数量限制 |
| **异常脱敏** | 错误信息不暴露本地路径和配置 |
| **部署友好** | 全部配置通过环境变量覆盖、支持生产/开发模式切换 |

---

## 📄 许可证

MIT License — 开源免费使用

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) — Agent 框架
- [Streamlit](https://github.com/streamlit/streamlit) — Web 应用框架
- [ChromaDB](https://github.com/chroma-core/chroma) — 向量数据库
- [Ollama](https://ollama.com) — 本地 LLM 运行环境

**参考开源项目：**
- swathiradhakrishnan06/Ecommerce-Q-A-Assistant-using-RAG（架构参考）
- ishan-dubey123/refund-agent（Agent模式参考）
- renaldiangsar/Customer-Support-RAG（PDF客服模式参考）
