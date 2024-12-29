# Seismic Arena
Comparative evaluation for seismic AI models

## TLDR;

Maybe we should just be [pairing off](https://lmsys.org/blog/2023-12-07-leaderboard/) attribute prediction models for geophysical tasks like fault detection and scoring the models relative to each-other. [I made a small app for that which you can find here](https://huggingface.co/spaces/porestar/seismic-arena). This is inspired by what is happening in LLM evaluations. 

## Introduction 

This application is inspired by the [LMSYS Chat Area](https://lmsys.org/blog/2024-03-01-policy/) to evaluate the outputs of large language models.  

How should we evaluate fault interpretations and attributes? Metrics often fail us.  
Maybe we should simply evaluate them based on ["vibes"](https://vickiboykis.com/2024/05/06/weve-been-put-in-the-vibe-space/) as we do for LLMs. 

The cool thing is that we can somewhat quantify "vibes" in terms of an [ELO Score](https://en.wikipedia.org/wiki/Elo_rating_system). By setting two models against each other in a battle we and letting a human rank the preference we can identify which models pass the vibe-check and put a number on it. 

In this case I use the exact implementation of the [scoring from the lmsys arena explained](https://lmsys.org/blog/2023-12-07-leaderboard/) [and demonstrated in this notebook.](https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH).

## Ranking Fault Segmentation models in the arena

To demonstrate this, I am sharing an open-source implementation of such an arena - technically this isn't anything new, someone just has to put a couple puzzle pieces together. 

The trigger for this was learning a bit more about [modal](https://modal.com) and specifically model deployments and the work done by [ProgrammerZXG](https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation) on cross-domain adaption of foundation models. 
Specifically the latter I found a) a number of datasets hosted on zenodo including one called deepfault, and b) it seems somewhat [potentially undertrained UNet implementations](https://github.com/ProgrammerZXG/Cross-Domain-Foundation-Model-Adaptation/blob/4217052de61422d1f55744c189bdab65c46a2083/dataset.py#L107) (it seems only left-right flipping as used with such a small dataset, these models quickly overfit). The datasets I have in the meanwhile ported onto [huggingface](https://huggingface.co/collections/porestar/seismic-foundation-model-datasets-67609032ab28896d0b256f55).

## Baselines

I have implemented some baselines and provide here notebooks, huggingface links, and mean iou scores on the validation dataset.

- [Full Augment](https://huggingface.co/porestar/deepfault-unet-baseline-full-augment) [Huggingface]() MeanIoU: 0.925
- [Weak Augment](https://huggingface.co/porestar/deepfault-unet-baseline-weak-augment) [Hugginface]()  MeanIoU: 0.802
- [No Augment](https://huggingface.co/porestar/deepfault-unet-baseline-no-augment) [Huggingface]()  MeanIoU: 0.72

To be fair, looking at the results of the models I am wondering if the validation set in this dataset is too similar to the training dataset which allows for overfitting not showing up as indications on the metrics, think of two images shifted by a single pixel, if I memorize the segmentations on one image it is likely I can predict the memorized segmentation and get a very good score. I will have to follow up with more investigation on this if my strong baselines are simply that good, or somethings fishy with the data.

## Implementing the arena

In ANY CASE, that is not my intent here, getting back to the vibe check.

I created a simple [gradio app](https://gradio.app/) that loads random images from the validation set, sends them to ML-inference endpoints hosted on [modal](https://modal.com/) which make the predictions. The user can then say which model is better and how they compare. Results are stored on Modal volumes. If you switch over to the leaderboard, the scores are computed and presented. 

The actual [front-end is hosted as a huggingface space]((https://huggingface.co/spaces/porestar/seismic-arena) and everything is orchestrated with github actions. 

## Disclaimer

This as was a learning project, intended as fun, to investigate and learn something new. It does raise a few things I would like to see more of: 
- host datasets on common easy access platforms with proper licensing
- host models on common easy access platforms and provide reproducible implementations
- provide strong baselines on common datasets and compare new methods against them
- think more about how we evaluate model outputs especially in times of stochastic models where 
no ground truth is available

## Final words

If you have questions feel free to reach out to me, raise issues, provide new endpoints, investigate the dataset, provide new datasets, rate models. 

Thank you to the folks at [modal](https://modal.com/) for providing me some credits to run these kinds of experiments. 

