import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

'''
    data info:
        corrects: array of size (500, 64), indicating whether each guess is correct
        probs: array of of size (500,) indicating the pass@64 rate for each sample
        lower: ???
        upper: ???
        n_corretcts:
        n_trials:
        pass@k: one floating number, proportions of problems covered with "best-of-k" sampling    
'''
# load data
alldata = json.load(open("alldata.json", "r"))
Ts = [0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
print(alldata.keys())

def acc_vs_steps():
    # steps
    pass_1_0 = [alldata[f"Qwen-1.5B_GRPO_{step}steps_T=0"]["pass@1"] for step in steps]
    pass_1 = [alldata[f"Qwen-1.5B_GRPO_{step}steps_T=1.0"]["pass@1"] for step in steps]
    pass_32 = [
        alldata[f"Qwen-1.5B_GRPO_{step}steps_T=1.0"]["pass@32"] for step in steps
    ]

    plt.plot(steps, pass_1_0, marker="o", label="Qwen-1.5B_GRPO_T=0 pass@1")
    plt.plot(steps, pass_1, marker="o", label="Qwen-1.5B_GRPO_T=1.0 pass@1")
    plt.plot(steps, pass_32, marker="o", label="Qwen-1.5B_GRPO_T=1.0 pass@32")
    plt.xticks(steps)
    plt.xlabel("Steps")
    plt.ylabel("Acc")
    plt.legend()
    plt.savefig("acc_vs_steps.png", dpi=300)

def acc_vs_temp():
    ''' deprecated '''
    n_samples = [1, 2, 4, 8, 16, 32, 64]
    cmap = plt.get_cmap("jet_r")
    n_sample_colors = {}
    for i_n_sample, n_sample in enumerate(n_samples):
        n_sample_colors[n_sample] = cmap(i_n_sample / len(n_samples))

    passes = []
    passes_rl = []
    for T in Ts:
        fn = f"Qwen-1.5B_T={T}"
        fn_rl = f"Qwen-1.5B_GRPO_95steps_T={T}"
        passes_T, passes_rl_T = [], []
        for i_n_sample, n_sample in enumerate(n_samples):
            key = f"pass@{n_sample}"
            if key in alldata[fn]:
                passes_T.append(alldata[fn][key])
            else:
                passes_T.append(np.nan)
            if key in alldata[fn_rl]:
                passes_rl_T.append(alldata[fn_rl][key])
            else:
                passes_rl_T.append(np.nan)
        passes.append(passes_T)
        passes_rl.append(passes_rl_T)
    passes = np.array(passes)
    passes_rl = np.array(passes_rl)
    passes.shape, passes_rl.shape

    plt.figure()
    for i_n_sample, n_sample in enumerate(n_samples):
        plt.plot(Ts, passes[:, i_n_sample], color=n_sample_colors[n_sample], ls="--")
        plt.plot(Ts, passes_rl[:, i_n_sample], color=n_sample_colors[n_sample], ls="-")
    for i_n_sample, n_sample in enumerate(n_samples):
        plt.plot(
            [],
            [],
            label="n_sample={}".format(n_sample),
            color=n_sample_colors[n_sample],
        )
    plt.plot([], [], ls="--", label="Pre-GRPO", c="black")
    plt.plot([], [], ls="-", label="Post-GRPO", c="black")
    plt.xlabel("Temperature")
    plt.ylabel("Pass")
    plt.legend(ncol=2)
    plt.xscale("log")
    plt.savefig("acc_vs_T.png", dpi=300)

    return

def solving_probs_temp():
    # Set style
    sns.set_theme(style="whitegrid", context="talk")

    # Temperatures to plot
    temperatures = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    n_temps = len(temperatures)

    # Prepare trimmed colormap (avoid white part at the top of 'hot')
    base_cmap = plt.get_cmap("gist_heat_r")
    trimmed_colors = base_cmap(np.linspace(0.25, 1.0, n_temps))  # cut off the brightest

    # Create figure with 2 shared-y subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = ["Qwen-1.5B", "Qwen-1.5B_GRPO_95steps"]

    for ax, model in zip(axes, model_names):
        for i, T in enumerate(temperatures):
            solve_probs = alldata[f"{model}_T={T}"]["probs"]
            x = np.arange(len(solve_probs))
            ax.plot(x, np.sort(solve_probs)[::-1], label=f"T={T}", color=trimmed_colors[i])

        ax.set_xlabel("Rank of Samples")
        ax.set_title("Post-GRPO" if "GRPO" in model else "Pre-GRPO")
        ax.set_ylim([-0.02, 1.02])
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel("Pass@64 Rate", fontsize=14)
    axes[1].legend(title="Temperature", fontsize=10)

    # on the third subplot, plot number of problems sovled with best-of-k sampling
    ax = axes[2]
    pre_grpo_num_solved = []
    post_grpo_num_solved = []   
    for i, T in enumerate(temperatures):
        num_solved = alldata[f"Qwen-1.5B_T={T}"]["pass@64"]*500
        pre_grpo_num_solved.append(num_solved)
        
        num_solved = alldata[f"Qwen-1.5B_GRPO_95steps_T={T}"]["pass@64"]*500
        post_grpo_num_solved.append(num_solved)


    ax.plot(temperatures, pre_grpo_num_solved, marker="o", label="Pre-GRPO")
    ax.plot(temperatures, post_grpo_num_solved, marker="o", label="Post-GRPO")
    ax.axhline(y=np.max(pre_grpo_num_solved), color="C0", linestyle="--")
    ax.axhline(y=np.max(post_grpo_num_solved), color="C1", linestyle="--")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("# of Problems Solved")
    ax.set_ylim([None, 500])
    ax.set_title("Best-of-N, N=64")
    ax.legend(title="Model", fontsize=10)

    plt.tight_layout()
    plt.savefig("solve_probs.pdf", dpi=300)

    return

def solving_probs_matched():
    # Set style
    sns.set_theme(style="whitegrid", context="talk")
    T = 1.0

    # Create figure with 2 shared-y subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    pre_grpo_solve_probs = np.array(alldata[f"Qwen-1.5B_T={T}"]["probs"])
    post_grpo_solve_probs = np.array(alldata[f"Qwen-1.5B_GRPO_95steps_T={T}"]["probs"])
    x = np.arange(len(pre_grpo_solve_probs))
    sort_idx = np.argsort(pre_grpo_solve_probs)[::-1]

    ax.plot(x, pre_grpo_solve_probs[sort_idx], color='C0', label="Pre-GRPO", ls="", marker="o", markersize=5)
    ax.plot(x, post_grpo_solve_probs[sort_idx], color='C1', label="Post-GRPO", ls="", marker="o", markersize=5)

    ax.set_xlabel("Test Problems")
    ax.set_ylim([-0.02, 1.02])
    ax.tick_params(labelsize=12)
    ax.set_ylabel("Pass@64 Rate", fontsize=14)
    plt.legend(title='T=1.0', fontsize=12)
    plt.savefig("solve_probs_matched.pdf", bbox_inches="tight")

    return

def subject_vs_acc():
    number_theory_idx = alldata["subject_inds"]["number"]
    probs_base = np.array(alldata["Qwen-1.5B_T=1.0"]["probs"])[number_theory_idx]
    subjects = list(alldata["subject_inds"].keys()) + ["all"]
    print("subjects:", subjects)
    probs_grpo_subjects = np.array(
        [np.array(alldata[f"Qwen-1.5B_GRPO_{subject}_T=1.0"]["probs"])[number_theory_idx] for subject in subjects]
    )
    sort_idx = np.argsort(probs_base)[::-1]
    # Train on 7 subject and eval on the entire test data
    fig, axes = plt.subplots(2, len(subjects)//2, figsize=(20, 10), sharey=True, sharex=True)
    axes = axes.flatten()

    for i, subject in enumerate(subjects):
        probs = probs_grpo_subjects[i]        
        axes[i].plot(
            np.arange(len(probs)), probs_base[sort_idx], marker="o", label="Pre-GRPO", ls="", color="C0"
        )
        axes[i].plot(
            np.arange(len(probs)), probs[sort_idx], marker="o", label=subject, ls="", color = f"C{i+1}"
        )
        axes[i].set_ylim([-0.02, 1.02])
        axes[i].tick_params(labelsize=12)
        axes[i].set_xlabel("Test Problems")
        axes[i].set_ylabel("Pass@64 Rate", fontsize=14)
    
        axes[i].legend()
    
    plt.savefig("subject_vs_acc_problem_matched.pdf", bbox_inches="tight")


    # Look at number_theory test performance afte training on different subjects
    # fig, axes = plt.subplots(2, len(subjects)//2, figsize=(20, 10), sharey=True, sharex=True)
    # axes = axes.flatten()
    # train_subj = 4 # randomly selected
    # for i, (key, subject_ind) in enumerate(alldata["subject_inds"].items()):
    #     base_probs_subject = probs_base[subject_ind]
    #     grpo_probs_subject = probs_grpo_subjects[train_subj][subject_ind]
    #     sort_idx = np.argsort(base_probs_subject)[::-1]

    #     axes[i].plot(
    #         np.arange(len(base_probs_subject)), base_probs_subject[sort_idx], marker="o", label="Pre-GRPO", ls="", color="gray"
    #     )
    #     axes[i].plot(
    #         np.arange(len(grpo_probs_subject)), grpo_probs_subject[sort_idx], marker="o", label=key, ls="", color = f"C{i+1}"
    #     )
    #     axes[i].set_ylim([-0.02, 1.02])
    #     axes[i].tick_params(labelsize=12)
    #     axes[i].set_xlabel("Test Problems")
    #     axes[i].legend()
    # plt.savefig("subject_vs_acc_subject_matched_2.png", dpi=300, bbox_inches="tight")


    


    ''' old '''
    # base_probs_per_subjects = []
    # probs_per_subjects = []
    # for key, subject_ind in alldata["subject_inds"].items():
    #     base_probs_per_subjects.append(probs_base[subject_ind].mean())
    #     probs_per_subjects.append(
    #         probs_grpo_subjects[:, subject_ind].mean(axis=-1)
    #     )  # remove all
    # base_probs_per_subjects = np.array(base_probs_per_subjects)
    # probs_per_subjects = np.array(probs_per_subjects)
    # base_probs_per_subjects.shape, probs_per_subjects.shape

    # plt.figure(figsize=(15, 5))
    # plt.subplot(121)
    # plt.imshow(probs_per_subjects)
    # cb = plt.colorbar(fraction=0.046, pad=0.04)
    # cb.set_label("Accuracy")
    # plt.xticks(range(len(subjects)), subjects, rotation=45)
    # plt.yticks(range(len(subjects[:-1])), subjects[:-1], rotation=45)
    # plt.xlabel("Training subject")
    # plt.ylabel("Test subject")
    # plt.subplot(122)
    # plt.imshow(probs_per_subjects - base_probs_per_subjects[:, None])
    # cb = plt.colorbar(fraction=0.046, pad=0.04)
    # cb.set_label("Delta Accuracy")
    # plt.xticks(range(len(subjects)), subjects, rotation=45)
    # plt.yticks(range(len(subjects[:-1])), subjects[:-1], rotation=45)
    # plt.xlabel("Training subject")
    # plt.ylabel("Test subject")
    # plt.savefig("acc_vs_subjects.png", dpi=300)

    return


# filepaths={
#     "Qwen-1.5B":"./data/new_evals_03_03/qwen-2.5-1.5b-instruct/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_algebra":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-algebra/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_algebra_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-algebra/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_counting":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-counting/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_counting_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-counting/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_geometry":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-geometry/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_geometry_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-geometry/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_interalgebra":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-interalgebra/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_interalgebra_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-interalgebra/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_number":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-number/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_number_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-number/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_prealgebra":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-prealgebra/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_prealgebra_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-prealgebra/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_precalculus":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-precalculus/math_500/temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_precalculus_t=0":"./data/new_evals_03_03/subjects/qwen-1.5b-grpo-math-precalculus/math_500/temp=0.0_n=1_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_all":"./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500//temp=1.0_n=32_ntokens=8192/data.json",
#     "Qwen-1.5B_GRPO_all_t=0":"./data/new_evals_03_03/qwen-2.5-1.5b-instruct-grpo-math-v1-130/math_500//temp=0.0_n=1_ntokens=8192/data.json",
# }
# names=["base","algebra","counting","geometry","interalgebra","number","prealgebra","precalculus","all"]
# labels=list(filepaths.keys())


if __name__ == "__main__":
    # acc_vs_steps()
    
    # acc_vs_temp())
    
    # solving_probs_temp()
    
    subject_vs_acc()

    # solving_probs_matched()
