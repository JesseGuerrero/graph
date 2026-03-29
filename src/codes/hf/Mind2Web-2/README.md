---
license: cc-by-4.0

configs:
- config_name: default
  data_files:
  - split: dev_set  
    path: "dev_set.csv"
  - split: test_set
    path: "test_set.csv"
---

# Mind2Web 2
 
Mind2Web 2 is an evaluation framework for agentic search capabilities, featuring Agent-as-a-Judge methodology for comprehensive assessment of web automation agents.
 
<div align="center">
<img src="https://github.com/OSU-NLP-Group/Mind2Web-2/blob/main/assets/mind2web2_overview.jpg?raw=true" alt="Mind2Web 2 Overview" width="800"/>
<p><em>Mind2Web 2 features realistic and diverse long-horizon web search tasks and a novel Agent-as-a-Judge framework to evaluate complex, time-varying, and citation-backed answers.</em></p>
</div>
 
## 🔗 Links
 
- [🏠 Homepage](https://osu-nlp-group.github.io/Mind2Web-2)
- [🏆 Leaderboard](https://osu-nlp-group.github.io/Mind2Web-2/#leaderboard)
- [📖 Paper](https://arxiv.org/abs/2506.21506)
- [💻 Code](https://github.com/OSU-NLP-Group/Mind2Web-2)

## 🔄 Changelog
- **Oct 23, 2025:** 
  - Updated several tasks to use dynamic relative time ranges instead of hardcoded time periods. 
  - All evaluation scripts are released for both public dev set and test set.
- Jun 27, 2025: Initial Release.

For details and old versions, please refer to [changelog.md](changelog.md).


## 📝 Citation Information
If you find this work useful, please consider citing our paper:
```
@inproceedings{
    gou2025mindweb,
    title={Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge},
    author={Boyu Gou and Zanming Huang and Yuting Ning and Yu Gu and Michael Lin and Botao Yu and Andrei Kopanev and Weijian Qi and Yiheng Shu and Jiaman Wu and Chan Hee Song and Bernal Jimenez Gutierrez and Yifei Li and Zeyi Liao and Hanane Nour Moussa and TIANSHU ZHANG and Jian Xie and Tianci Xue and Shijie Chen and Boyuan Zheng and Kai Zhang and Zhaowei Cai and Viktor Rozgic and Morteza Ziyadi and Huan Sun and Yu Su},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=AUaW6DS9si}
}
```
