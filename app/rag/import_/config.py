from app.shared.config.common import env_str

MINERU_MODEL_VERSION = env_str("MINERU_MODEL_VERSION", "vlm")
# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 外部服务排队和解析耗时不固定，不将此上限当作单篇解析速度。
MINERU_POLL_TIMEOUT_SECONDS = 600

# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3
MINERU_REQUEST_TIMEOUT_SECONDS = 30

# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30


# 定义local_dir对应输出的常量
PDF_PARSE_SERVICE_LOCAL_DIR = "output"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 章节进入细分的长度阈值，同时是普通文本合并上限（字符）；结构块另行保真。
CHUNK_MAX_SIZE = 1000

# 超过上限的普通长文细分目标（字符，不是token）；不是启动切分的阈值。
CHUNK_SIZE = 600

# 普通长文细分时重叠目标100字符；并非每相邻块都恰好重叠100。
# 2026-10-16真实6份资料/11组对比后采用，证据见output/day03/chunk_benchmark。
CHUNK_OVERLAP = 100

# 最小碎片阈值: 低于这个长度判定为短碎片, 需要尝试合并
CHUNK_MIN = 400

# 声明常量，切块的截取数量 默认： 10
CHUNKS_SPLIT_TOP_NUMBER = 10

# 向量化批次大小：每批处理 6 条切片；资源占用仍需按实际长度和设备验收
EMBEDDING_BATCH_SIZE = 6
