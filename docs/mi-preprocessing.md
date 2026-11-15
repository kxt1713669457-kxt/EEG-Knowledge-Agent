# 运动想象原始EEG预处理

本模块将连续运动想象记录变成可追溯的数值试次，保留来源、事件、处理版本和质量报告。后续分类仍须符合对应权重的独立输入契约。
数值处理在Python中执行，Agent只选择已登记方案和解释真实产物。

## 使用入口

现在可直接在聊天输入框点击“脑电数据”或拖入文件，然后发送给Agent研究，无需打开登记窗口。
上传只暂存并核验文件头；正式登记、预处理和推理由Agent按消息选择工具执行。

- 连续原件示例：“这是BCI IV 2a原始GDF，用DWDaNet原流程处理前2个试次，再识别第1个试次（下标0）。”
- 已切段样本示例：“这是BCI IV 2a样本，250 Hz、单位V，数组轴序为[试次,通道,采样点]，只帮我登记，不识别。”
- 信息不足时Agent先检查附件并请求采样率、单位或事件协议；附件保留供下一轮补充，换会话会清空选择。

本轮最多5份、合计128MiB，支持GDF/EDF/BDF/FIF/连续NPZ以及试次MAT/NPY/NPZ。
SHU原流程须同时选择同一被试全部5个会话MAT。附件以真实ID绑定登记、预处理和推理结果，不接受Agent传入本地路径。
原有窗口仍可手动使用，入口如下。

启动项目原有查询服务后，在聊天页侧栏选择“运动想象预处理”。

1. 上传一份GDF/EDF/BDF/FIF，选择BCI IV 2a、2b或通用MI。单文件上限128MiB。
2. 2a/2b先选“原模型协议 · 提示后4秒”；其他原采样率选独立“原生采样率”方案，默认32试次。一次只处理一份记录。
3. 检查处理前后质量指标、被拒绝窗口和事件位置，下载JSON报告/NPZ。
4. 点击“使用本次数值样本”，选择已训练识别模型，再发送分析请求。预处理自身不执行分类。
5. 刷新“已有预处理产物”可以读取旧结果，不重复处理。

原始GDF中的EOG可能被读取库默认识别成EEG，本模块会按真实EOG类型/名称显式排除。
已有试次MAT/NPY/NPZ继续走原“登记样本”入口；连续原件和试次数组不能混用。

原有七个方案保持250Hz、提示后4秒、1000点的限制。新增三个`sampling_policy=native`独立方案保留100–1000Hz原采样率和真实EEG通道，4秒对应`4*sfreq`点（须为整数），不进行重采样。
例如PhysioNet EEGMMIDB的64通道、160Hz原件输出`[trial,64,640]`，不能接现有22通道、250Hz、1000点分类权重。原始64通道不会被裁成22通道，也不会补造分类头。

## 已实现方案

另已分别登记 `dwdanet-original-v1`、`dwtnet-original-v1`，名称为“我的原数据处理流程”。
两套来源与训练配方独立记录，但2a/2b信号切段和SHU整理使用共同的原协议；不虚构不同滤波算法。
具体原文件、SHA256、归一化公式和评估范围见`original_workflows/dwdanet.json`与`dwtnet.json`。

选择SHU-MI时，一次上传同一被试session1至5的五份`sub-XXX_ses-YY_task_motorimagery_eeg.mat`。
它们已经是作者切好的32通道四秒试次，按明确的两种原划分之一处理：

- `shu-session1to4-vs5-v1`：session1–4训练、5测试。本地C01/C02已核对符合此版。
- `shu-mixed-session-stratified70-v1`：五session混合后按类分层70/30，RandomState种子`2026+subject_id`；不代表跨session泛化。

两版均严格保留原float32数值与1/2标签，导出`train.mat/test.mat`，单位仍为µV；仅测试集前N试次
转换为V并登记为推理预览样本。标签只用于数据划分和导出，不传入模型；不覆盖用户原Cxx文件。
归一化配方保留在流程契约中，本预处理不重新拟合或提前应用mean/std，避免与模型适配器重复。
目前可执行分类权重仅核验2a；没有把2b或SHU的输入整理成功说成已训练/验证对应识别模型。

| ID | 处理 | 与当前权重关系 |
|---|---|---|
| `bci-cue4s-legacy-v1` | 保持原参考/EEG顺序，无额外软件滤波、重采样或去均值 | 对照现有2a/2b处理脚本；2a真实MAT已逐值验证 |
| `mi-mu-beta-v1` | 连续有效片段8–32Hz、四阶Butterworth双向滤波 | 研究对照，未证明提高分类效果 |
| `mi-broadband-v1` | 连续有效片段4–40Hz、四阶Butterworth双向滤波 | 研究对照 |
| `mi-eog-regression-v1` | 平均参考、真实EOG回归、4–40Hz | 改变原参考；需要真实EOG |
| `mi-pyprep-qc-v1` | PyPREP坏道检测，仅标记，不修改信号；禁用RANSAC | 至少16通道；不自动插值/删除 |
| `mi-autoreject-qc-v1` | AutoReject阈值/坏试次日志，不剔除或插值 | 至少12试次、16通道及真实位置；单CPU、CV=3 |
| `mi-iclabel-v1` | 平均参考、1–100Hz、extended Infomax、ICLabel、4–40Hz | 至少16通道及真实位置；至多32试次、120秒拟合、15分量；仅移除得分≥0.90的眼/肌/心电成分 |
| `mi-native-cue4s-v1` | 原采样率下提示后4秒；保留全部EEG、原参考，无额外处理 | 通用研究试次；不能据此宣称适配现有权重 |
| `mi-native-mu-beta-v1` | 原采样率下连续有效片段8–32Hz、四阶双向滤波 | 与原生不滤波方案独立比较，不重采样 |
| `mi-native-broadband-v1` | 原采样率下连续有效片段4–40Hz、四阶双向滤波 | 同上；160Hz输出640点，64通道保持64通道 |

实际依赖就绪情况由目录显示；具体记录不满足条件时明确拒绝执行。
3通道2b不默认使用空间插值或ICA，输出3通道也不能接现有22通道权重。
匿名通道如`EEG-0`不能凭通道总数猜成标准电极；需要有来源的显式映射或坐标。
通用MI数据即使形状相同，也不表示已验证与2a训练域一致。

新增处理方案可能改变模型输入分布。应先与原协议比较，必要时重新训练；质量指标下降不等于分类准确率上升。
所有方案均为离线研究；双向滤波不能称为实时因果处理。记录缺口和采集断点不跨段滤波，跨断点的试次被拒绝。

## 连续NPZ与补充元信息

无原生读取器的连续数据可以提供以下非对象数组（NPY v1/v2）：

```python
np.savez_compressed(
    'recording.npz',
    signal=signal,              # [channel, time]连续实数矩阵
    sfreq=np.array(250.0),
    ch_names=np.array(names),   # 唯一Unicode通道名
    events=events,              # 整数[N,3]: [原始采样位置, 0, 原始事件码]
    unit=np.array('V'),         # V或uV，不按振幅猜单位
)
```

2a原始cue码为769/770/771/772，评估未知cue为783；2b使用769/770和783。
MNE自动生成的内部事件编号不等于原始GDF码，本模块会显式保留原始码。
783只用于找窗口，不能据此推断类别，不加载`classlabel`或用任务标签去拟合处理。

通用事件码、EOG名和标准电极映射通过上传框的JSON明确声明：

```json
{"cue_event_codes":[1,2],"eog_channels":["EOG-left"],"channel_mapping":{"original-C3":"C3"}}
```

`channel_positions`可提供实际米制头坐标`{"C3":[x,y,z]}`，不能伪造位置。
上述mapping仅为格式示例，需要替换成来源已核对的真实映射。

原生EDF/FIF等记录若使用字符串注释，需显式声明注释映射，不能使用MNE自动分配的内部编号代替来源事件：

```json
{"annotation_event_map":{"T1":1,"T2":2},"cue_event_codes":[1,2]}
```

`annotation_event_map`最多32项，注释名须为1–80字符，值为0–65535整数（不接受bool）。连续NPZ已有整数`events`，不再接受字符串注释映射。
对PhysioNet EEGMMIDB的左/右拳**运动想象**运行R04/R08，上例映射仅用于这两个明确任务；不能将其他run中的T1/T2无条件解释为左/右手。通道名使用`corpus/eeg/datasets/external_raw/physionet_eegmmidb/manifest.json`中核对的完整映射，保留64个实录EEG通道。EDF校准由MNE转换成V，不再次乘微伏系数。通道位置属于标准模板，不冒称被试实测位置。

## 可复现性与资源

- 原件不可变保存于`output/runtime/mi-preprocessing/recordings`。
- 产物在`output/runtime/mi-preprocessing/artifacts/<id>`，含试次NPZ、JSON；ICA方案另存完整ICA FIF状态。
- 原始哈希、执行代码摘要、库版本、事件原始位置、参数、拟合范围、系数/成分和QC均留档。
- 输出为V；冻结归一化由各模型适配器执行一次，不在预处理里重建训练统计。
- 派生样本与源产物核对身份/哈希后才能进入模型。
- 连续记录上传128MiB、解码256MiB、输出96MiB上限；SHU五会话合计上传128MiB、处理内存估计512MiB上限。内存不足拒绝，单记录串行、CPU执行。
- 测试均使用临时目录，验收默认只保留紧凑报告；`--persist`才保留工作台样本。
- 本地预处理和离线验收不需要Mongo/MinIO/Milvus，因此不启动Docker虚拟机或无关服务。

安装：`uv sync --extra mi-preprocessing`。MNE为核心依赖，其余质检/ICLabel为可选扩展。

## 验证

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .\.venv\Scripts\python.exe -m pytest tests/test_mi_preprocessing.py tests/test_mi_preprocessing_integration.py tests/test_mi_native_sampling.py -q
node scripts/verify_mi_preprocessing_ui.cjs
& .\.venv\Scripts\python.exe scripts/verify_mi_preprocessing.py --raw 'A01T.gdf' --reference-mat 'A01T.mat' --trials 16 --classify
```

真实验收脚本不会从服务器读取任意路径；路径仅属于本机CLI用户输入。HTTP/Agent只接受已登记ID。
现有模型manifest中的归一化是重建统计，不冒称找回原训练时保存的精确统计或复现论文准确率。

## 新方法调查与来源

以下候选独立于可执行方案，目录标为`research_only`，不自动下载大权重：

- [EEGOAR-Net（2025）](https://github.com/dmarcos97/EEGOAR-Net)：有作者权重，但128Hz、1秒、64模板和掩码输入需要单独适配；尚未实现。
- [FAAR（2026）](https://arxiv.org/html/2605.12408v1)：面向MI的自适应质量筛选；未核到公开作者实现，不以自写简化评分冒充复现。
- [ASR/MEEGkit](https://nbara.github.io/python-meegkit/modules/meegkit.asr.html)：需校准参考、秩与阈值验证，尚未启用。
- [MIRepNet](https://braindecode.org/stable/generated/braindecode.models.MIRepNet.html)：表征/分类与分布对齐路线，非通用去伪迹器。

已实现工具的官方依据：
[MNE](https://mne.tools/stable/api/preprocessing.html)、
[MOABB运动想象频段](https://moabb.neurotechx.com/docs/generated/moabb.paradigms.MotorImagery.html)、
[EOG回归](https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html)、
[PyPREP](https://pyprep.readthedocs.io/en/stable/generated/pyprep.NoisyChannels.html)、
[AutoReject](https://autoreject.github.io/stable/generated/autoreject.AutoReject.html)、
[ICLabel输入条件](https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html)、
[BCI IV 2a原始协议](https://www.bbci.de/competition/iv/desc_2a.pdf)。

## 不同真实通道数验收（2026-11-12）

| 原始记录 | 实际EEG通道 | 原采样率 | 本次原样输出 | 检查 |
|---|---:|---:|---|---|
| BCI IV 2b `B0201T.gdf` | 3（另有3 EOG） | 250Hz | 16×3×1000 | 与既有MAT对应窗口逐值相同；4种方案实际执行 |
| BCI IV 2a `A02T.gdf` | 22（另有3 EOG） | 250Hz | 16×22×1000 | 与既有MAT对应窗口逐值相同；5种方案实际执行 |
| PhysioNet `S001R04.edf` | 64，无EOG | 160Hz | 15×64×640 | 原样窗口逐值相同；3种native方案实际执行 |
| PhysioNet `S001R08.edf` | 64，无EOG | 160Hz | 15×64×640 | 同上；独立的第二个运动想象运行记录 |

PhysioNet两份EDF来自[官方数据入口](https://physionet.org/content/eegmmidb/1.0.0/)，
共约4.95MiB，SHA256与官网清单一致。许可为[ODC-By 1.0](https://physionet.org/content/eegmmidb/view-license/1.0.0/)。
T1/T2分别是想象左/右拳开合，T0是休息；仅选择T1/T2提窗。原件和来源/注释清单保存在
`corpus/eeg/datasets/external_raw/physionet_eegmmidb`。

这里的“原始”指公开发布的连续记录、未额外软件处理或切窗；不保证采集硬件未滤波。
2a采集滤波见官方协议；PhysioNet EDF未声明完整硬件滤波/原参考，MNE的80Hz值为Nyquist默认上限，
不能当成已核实的80Hz采集低通。完整范围在下载manifest中说明。

未将32通道SHU列为原始连续验收：其公开数据已坏段剔除、去基线、0.5–40Hz滤波并保留4秒试次；
公开EDF也是这些保留窗的拼接。[作者数据描述](https://www.nature.com/articles/s41597-022-01647-1)、
[MOABB分发说明](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ma2022.html)。

验收报告：`output/validation/mi-a02t-real.json`、`mi-b0201t-real.json`、`mi-native64-real.json`。
它们验证实际执行、来源和切段一致性，不是新预处理提高分类率的结论。

## 两套原流程与界面验收（2026-11-12）

- `output/validation/mi-original-workflows-real.json`：两套family分别完成2a、2b、SHU两种划分，共8例；2a/2b及SHU会话1–4/5划分与原MAT逐值一致。测试临时副本已清理。
- `output/validation/mi-final-regression.txt`：244项测试、114个subtest通过。
- `output/validation/mi-ui-checks.json`：24项界面逻辑检查通过；此外实际登录工作台上传SHU五会话、执行原流程并下载测试集。
- `output/validation/mi-shu-browser-download.json`：浏览器下载MAT的SHA256与产物报告一致，测试集94×32×1000、训练集398×32×1000；下载产生的重复副本已删除，保留工作台正式产物。
- `output/validation/screenshots/mi-shu-original-result.jpg`、`mi-shu-original-export.jpg`、`mi-64ch-result.jpg`：实际界面截图。

工作台演示产物保留在“已有预处理产物”，读取不会重复计算。SHU32通道及PhysioNet64通道没有兼容且就绪的已核验分类权重时，界面禁用分类入口，仍允许下载处理结果。

## Agent能力分组

“Skills能力”按预处理Skills、推理Skills展示。预处理方案不会进入推理模型选择下拉。
预处理主说明位于`app/agent/skills/mi_preprocessing/SKILL.md`，推理主说明位于`app/agent/skills/eeg_inference/SKILL.md`。
Agent涉及这些数据任务时加载两类说明；各模型权重、输入契约与原流程JSON保留独立来源。
仅检查和登记的回执由实际工具元信息直接生成，无需再调用模型查文献或猜测字段。

聊天接入验收记录见`output/validation/mi-agent-chat-real.json`：真实A02T GDF经聊天上传后，
登记、DWDaNet原流程预处理、DWDaNet识别三步成功，前2个试次与原MAT逐值相同；
另验证缺少单位/采样率时先询问，补充后只登记，再在下一轮以DWTNet识别下标1而不重复登记/预处理。
刷新页面后附件名称和结果历史仍保留。完整回归262项及114个subtest通过；
针对阶段复用规则的后续专项41项及10个subtest通过，附件UI测试、24项预处理UI检查及两份技能格式校验通过。
