"""Deterministic receipts for attachment inspection and registration-only tasks."""


def operation_receipt(execution):
    trace = execution.get("trace") or []
    allowed = {"inspect_eeg_attachment", "register_eeg_recording", "register_eeg_sample"}
    if not trace or any(step.get("tool") not in allowed for step in trace):
        return None
    lines = []
    for result in execution["tool_results"].values():
        if result["status"] != "ok":
            lines.append("未完成：" + (result.get("error") or {}).get("message", "附件操作未完成。"))
            continue
        data = result["data"]
        row = data.get("registration")
        if row:
            if row.get("kind") == "sample":
                lines.append(f"已登记数值样本：{row['file']}。样本ID：{row['sample_id']}；形状：{' × '.join(map(str, row['shape']))}；{row['sampling_rate_hz']} Hz；单位 {row['unit']}。")
                if row.get("compatibility_note"):
                    lines.append(row["compatibility_note"])
            else:
                lines.append(f"已登记脑电记录：{row['file']}。记录ID：{row['recording_id']}；{row['channels']}通道；{row['sampling_rate_hz']} Hz；数据集 {row['dataset_id']}。")
            continue
        row = data.get("attachment")
        if row:
            header = row["header_summary"]
            shape = header.get("shape")
            details = ("；数组形状 " + " × ".join(map(str, shape))) if shape else (f"；文件头包含 {header['channels_including_non_eeg']} 个通道" if header.get("channels_including_non_eeg") else "")
            lines.append(f"已读取附件：{row['filename']}{details}。")
            if row["representation"] == "epoched":
                lines.append("这是待核验的已切段试次数组。请提供采样率（例如250 Hz）、数值单位（V或uV），并确认数据集和数组轴序为[试次,通道,采样点]或[通道,采样点]，我就能继续登记。")
            else:
                lines.append(f"连续记录的采样率为 {header.get('sampling_rate_hz')} Hz。请确认数据集及运动想象事件协议；通用数据还需明确提示事件码或注释映射，不能凭文件名猜测。")
    registration = any(result["status"] == "ok" and result["data"].get("registration") for result in execution["tool_results"].values())
    lines.append("本次仅完成登记，尚未执行预处理或模型识别。" if registration else "本次只检查附件，尚未登记样本、预处理或执行模型。")
    return "\n\n".join(lines)
