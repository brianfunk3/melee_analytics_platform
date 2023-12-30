# Melee Analytics Platform
*this is where I'd put a cool logo or something if I were talented*

### Goal
The Melee Analytics Platform (MAP? Sure why not) will be a tool for in-depth analytics on competitive Super Smash Brothers Melee matches. Using computer vision models for detection and recognition, MAP will be able to track a match's clock, player damages, and even character locations over the course of a competitive match. These datapoints will be the source for post-match charts and insights that are automatically generated when a match is recognized. 

### Why?
Honestly because it'll be really cool. Doesn't seem like anyone has done anything in this space and I think it be a fun test of all of my skills. The post-match analytics I've seen so far haven't been very informative so lets see if we can fix that 🫡

### Status
Holy spinolli this is about as in-progress as it gets. Still have so much code to write and models to train, so don't expect meaningful outputs any time soon

### Repo Notes
- This repo will home to code only. Meaning no `.csv` files, no images/annotations, and certainly no models. However, this repo will house all code used to create those outputs. Moaybe someday that could change but this repo will basically be the skeleton for a pipeline and not much else.
- I work in Jupyter Lab like a sicko, so all functions are run in notebooks. But those functions are in `.py` files so people can run them however they'd like.
- I really should be using pull requests and branches but I won't since I'm the only one working here and things are still in testing mode. That will change once things are developed more though