def add_remaining_useful_life(df):  # 定义计算剩余使用寿命的函数，输入为数据框df
    # 获取每个单位总周期数
    grouped_by_unit = df.groupby(by="unit_nr")  # 按照unit_nr分组数据
    max_cycle = grouped_by_unit["time_cycles"].max()  # 获取每个单位的最大周期数

    # 将最大周期数合并回原始数据框
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)  # 合并最大周期列

    # 计算每行的剩余有效寿命（分段线性）
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]  # 计算剩余使用寿命
    result_frame["RUL"] = remaining_useful_life  # 将剩余使用寿命添加为新列

    # 删除max_cycle列，因为它不再需要
    result_frame = result_frame.drop("max_cycle", axis=1)  # 删除合并后的max_cycle列

    return result_frame  # 返回更新后的数据框
