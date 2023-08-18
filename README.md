# Latent Jailbreak

This repository contains the code and data for the paper **Latent Jailbreak: A Test Suite for Evaluating Both Text Safety and Output Robustness of Large Language Models**. The paper explores the topic of _latent jailbreak_ and presents a novel approach to evaluate the text safety and output robustness for large language models.

## Data

The data used in this paper is included in the `data` directory.

## Templates

![templates](./img/latent-jailbreak-dataset.png)
Templates for latent jailbreak prompts.

## Generate Model Responses

```bash
cd src
python BELLE_7B_2M.py
python ChatGLM2-6B.py
python ChatGPT.py --api_key 'your key'
```

## Fine-Tune Model to Perform Automatic Labeling

```bash
python finetune.py
```

## Citation

If you use the code or data in this repository, please cite the following paper.

```
@misc{qiu2023latent,
      title={Latent Jailbreak: A Test Suite for Evaluating Both Text Safety and Output Robustness of Large Language Models},
      author={Huachuan Qiu and Shuai Zhang and Anqi Li and Hongliang He and Zhenzhong Lan},
      year={2023},
      eprint={2307.08487},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
