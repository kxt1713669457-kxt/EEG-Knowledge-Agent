---
name: dwtnet-single-trial
description: 对输入契约兼容的已登记2a数值试次运行本地已训练dwtnet分类模型；不用于仅有波形截图的类别判定。
---

通过analyze_eeg_sample调用model_id=dwtnet-2a-local，只接受真实sample_id和从0计数的trial_index。
使用本机目录中的固定训练权重、标签和归一化参数，返回artifact_id、模型版本、真实logits/softmax与预测。
本地适配为22通道、250Hz、1000采样点、V；uV输入先换为V，不自动补通道或将截图反演成数值。
将结果表述为“该模型本次预测”，softmax未经校准，不是正确率或人工真值。归一化从原训练源目录重建，适用域和协议差异随产物说明交付。
追问已有结果使用get_eeg_analysis读取artifact_id；停用/移除skill后不再执行，但原产物仍可读。
