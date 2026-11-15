---
name: motor-imagery-inference
description: 将聊天上传的已切段数值脑电登记为样本，并调用输入兼容的已训练运动想象模型或表征编码器，解释真实执行结果。
---

已切段MAT/NPY/NPZ通过register_eeg_sample登记；只使用本轮真实attachment_id。
必须明确采样率、V/uV单位和数据集；轴序为[试次,通道,采样点]或单试次[通道,采样点]。
文件头没有单位/采样率时先inspect_eeg_attachment并向用户询问，不能由幅值、通道数、文件名推断。
用户仅要求登记时只返回实际sample_id、形状和可用模型，不运行推理。

已登记样本使用analyze_eeg_sample(sample_id,trial_index,model_id)，试次从0计数。
新登记样本的同一计划用registration_step_id绑定前置登记步骤，不能猜未来sample_id。
原始连续记录须先由预处理skill切段，之后analyze_preprocessed_eeg以preprocessing_step_id绑定真实前置产物。
若已取得artifact_id，追问解释使用get_eeg_analysis或get_eeg_preprocessing读取，不重复计算。

当前已核验分类模型dwdanet-2a-local、dwtnet-2a-local均要求22通道、250Hz、1000点，与2a训练协议相符。
3通道2b、32通道SHU、64通道160Hz不能直接使用这些权重；不裁补通道、不凭形状兼容宣称跨域有效。
CBraMod仅输出表征，没有本地已训练MI分类头，不能提供左右手预测。
推理读取模型固定权重及归一化，uV输入转换V；不从待分析样本拟合训练统计，不重复归一化。

回答给出本次模型预测、试次索引、模型/权重版本和可追溯artifact_id。
softmax为未经校准的类别分数，不是正确率或真值。单试次预测不能冒充整份数据准确率；
没有真实标签及完整评估时不编造准确率、混淆矩阵或预处理带来的提升。
模型细节按需读取[DWDaNet](../dwdanet/SKILL.md)、[DWTNet](../dwtnet/SKILL.md)。
