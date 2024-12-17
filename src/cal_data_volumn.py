# encoding: utf-8
"""
This script calculates the data volume and ratios for two stages of a project.

Stage I: Bilingual Adaptation
Stage II: Synthetic Enhancement

Variables:
    stage1_volumn (float): Volume for Stage I.
    stage2_volumn (float): Volume for Stage II.
    first_stage (list): Ratios for Chinese (cn) and English (en) in Stage I.
    second_stage (list): Ratios for Chinese (cn), English (en), and Synthetic (synth) in Stage II.
    cn_ratio (list): Ratios for different Chinese data sources.
    en_ratio (list): Ratios for different English data sources.

Calculations:
    stage1_ratio_lst (list): Calculated ratios for Stage I based on `first_stage` and `cn_ratio`/`en_ratio`.
    stage2_ratio_lst (list): Calculated ratios for Stage II based on `second_stage` and `cn_ratio`/`en_ratio`.
    data (list): Calculated data volumes for each data source.
    data_ratio (list): Calculated data ratios as percentages.

Outputs:
    Prints the calculated ratios for Stage I and Stage II.
    Prints the sum of the ratios for Stage I and Stage II.
    Prints the calculated data volumes for each data source.
    Prints the sum of the data volumes.
    Prints the calculated data ratios as percentages.
    Prints the sum of the data ratios as percentages.
"""

if __name__ == "__main__":
    print("Calculating Data Volume and Ratios for Two Stages")

    stage1_volumn = 92.5  # Stage I: Bilingual Adpatation
    stage2_volumn = 7.5  # Stage II: Synthetic Enhancement

    first_stage = [
        0.2,  # cn
        0.8,  # en
    ]
    second_stage = [
        0.1,  # cn
        0.7,  # en
        0.2,  # synth
    ]

    cn_ratio = [
        0.7,  # web-cn
        0.05,  # encyclopedia-cn
        0.2,  # book-cn
        0.05,  # qa_forum-cn
        0.0,
        0.0,
        0.0,
    ]
    en_ratio = [
        0.4,  # web-en
        0.05,  # encyclopedia-en
        0.15,  # book-en
        0.05,  # qa_forum-en
        0.1,  # paper-en
        0.1,  # math-en
        0.15,  # code-en
    ]

    # Calculate the data ratio for the two stages
    stage1_ratio_lst = [
        first_stage[0] * cn_ratio[0],
        first_stage[1] * en_ratio[0],
        first_stage[0] * cn_ratio[1],
        first_stage[1] * en_ratio[1],
        first_stage[0] * cn_ratio[2],
        first_stage[1] * en_ratio[2],
        first_stage[0] * cn_ratio[3],
        first_stage[1] * en_ratio[3],
        first_stage[1] * en_ratio[4],
        first_stage[1] * en_ratio[5],
        first_stage[1] * en_ratio[6],
    ]

    stage2_ratio_lst = [
        second_stage[0] * cn_ratio[0],
        second_stage[1] * en_ratio[0],
        second_stage[0] * cn_ratio[1],
        second_stage[1] * en_ratio[1],
        second_stage[0] * cn_ratio[2],
        second_stage[1] * en_ratio[2],
        second_stage[0] * cn_ratio[3],
        second_stage[1] * en_ratio[3],
        second_stage[1] * en_ratio[4],
        second_stage[1] * en_ratio[5],
        second_stage[1] * en_ratio[6],
        second_stage[2],
    ]

    stage1_ratio_lst = [round(r, 3) for r in stage1_ratio_lst]
    stage2_ratio_lst = [round(r, 3) for r in stage2_ratio_lst]

    print("Stage 1 Ratios: ", stage1_ratio_lst)
    print("Stage 2 Ratios: ", stage2_ratio_lst)
    print("Sum of Stage 1 Ratios: ", sum(stage1_ratio_lst))
    print("Sum of Stage 2 Ratios: ", sum(stage2_ratio_lst))

    data = []

    for cn_r, en_r in zip(cn_ratio, en_ratio):
        data.append(
            stage1_volumn * first_stage[0] * cn_r
            + stage1_volumn * first_stage[1] * en_r
            + stage2_volumn * second_stage[0] * cn_r
            + stage2_volumn * second_stage[1] * en_r
        )
    data.append(stage2_volumn * second_stage[2])
    data = [round(d, 5) for d in data]

    print("Data Volume List: ", data)
    print("Sum of Data Volumes: ", sum(data))

    data_ratio = [d / sum(data) * 100 for d in data]
    data_ratio = [round(d, 2) for d in data_ratio]

    print("Data Ratios (%): ", data_ratio)
    print("Sum of Data Ratios (%): ", sum(data_ratio))
