# PaoDing

An opiniated NLP-oriented PyTorch wrapper that makes your life easier. It is in spirit similar to AllenNLP. The goal of this library is to hide as much boilerplate code as possible, while still maintaining large control over my experiments. Be careful if you plan to depend on this library directly. It is intended to be a personal infrastructure library, so while I should generally be happy to fix bugs, I may not have the bandwidth to implement feature requests if it's not directly related to what I'm doing. Though PRs are always welcome.

## Examples

The below examples trains and evaluate on MNLI. In our environment, it achieves 86.5%/86.6% accuracy on MNLI matched/mismatched dev sets.

```bash
python examples/train_sequence_classification.py --data_dir data_cache --transformer_model bert-large-cased --batch_size 32 --max_length 256 --lr 0.00001 --warmup_ratio 0.06 --epochs 3 --clip_norm 1.0 --output_dir mnli
python examples/evaluate.py --ckpt_path mnli/best.ckpt
```

For a more realistic example, see https://github.com/ZhaofengWu/transparency.

## Etymology

Pao Ding (庖丁) is a character in the classic ancient Chinese text [Zhuangzi](https://en.wikipedia.org/wiki/Zhuangzi_(book)), published around the 3rd century BC. Pao (庖) means cook, his occupation, and Ding (丁) is his name.

> 庖丁为文惠君解牛，手之所触，肩之所倚，足之所履，膝之所踦，砉然向然，奏刀𬴃然，莫不中音；合于《桑林》之舞，乃中《经首》之会。
>
> 文惠君曰：「嘻，善哉！技盖至此乎？」
>
> 庖丁释刀对曰：「臣之所好者道也，进乎技矣。始臣之解牛之时，所见无非全牛者。三年之后，未尝见全牛也。方今之时，臣以神遇而不以目视，官知止而神欲行。依乎天理，批大郤导大窾因其固然。枝经肯綮之未尝微滞，而况大軱乎！良庖岁更刀，割也；族庖月更刀，折也。今臣之刀十九年矣，所解数千牛矣，而刀刃若新发于硎。彼节者有间，而刀刃者无厚；以无厚入有间，恢恢乎其于游刃必有余地矣。是以十九年而刀刃若新发于硎。虽然，每至于族，吾见其难为，怵然为戒，视为止，行为迟。动刀甚微，謋然已解，牛不知其死也，如土委地。提刀而立，为之四顾，为之踌躇滿志，善刀而藏之。」
>
> 文惠君曰：「善哉！吾闻庖丁之言，得养生焉。」

陈鼓应. 庄子今注今译[M]. 北京：中华书局，2016:106-107.

> Cook [D]ing was cutting up an ox for Lord Wen-hui. At every touch of his hand, every heave of his shoulder, every move of his feet, every thrust of his knee - zip! zoop! He slithered the knife along with a zing, and all was in perfect rhythm, as though he were performing the dance of the Mulberry Grove or keeping time to the Ching-shou music.
>
> "Ah, this is marvelous!" said Lord Wen-hui. "Imagine skill reaching such heights!"
>
> Cook [D]ing laid down his knife and replied, "What I care about is the Way, which goes beyond skill. When I first began cutting up oxen, all I could see was the ox itself. After three years I no longer saw the whole ox. And now - now I go at it by spirit and don't look with my eyes. Perception and understanding have come to a stop and spirit moves where it wants. I go along with the natural makeup, strike in the big hollows, guide the knife through the big openings, and follow things as they are. So I never touch the smallest ligament or tendon, much less a main joint.
>
> "A good cook changes his knife once a year-because he cuts. A mediocre cook changes his knife once a month-because he hacks. I've had this knife of mine for nineteen years and I've cut up thousands of oxen with it, and yet the blade is as good as though it had just come from the grindstone. There are spaces between the joints, and the blade of the knife has really no thickness. If you insert what has no thickness into such spaces, then there's plenty of room - more than enough for the blade to play about it. That's why after nineteen years the blade of my knife is still as good as when it first came from the grindstone.
>
> "However, whenever I come to a complicated place, I size up the difficulties, tell myself to watch out and be careful, keep my eyes on what I'm doing, work very slowly, and move the knife with the greatest subtlety, until - flop! the whole thing comes apart like a clod of earth crumbling to the ground. I stand there holding the knife and look all around me, completely satisfied and reluctant to move on, and then I wipe off the knife and put it away."
>
> "Excellent!" said Lord Wen-hui. "I have heard the words of Cook [D]ing and learned how to care for life!"

Watson, Burton, trans. 1964. Chuang Tzu, basic writings. New York: Columbia University Press.
