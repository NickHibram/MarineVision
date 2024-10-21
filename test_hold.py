from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import requests

import torch
import os
import glob
import time
import statistics
import numpy as np


model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
texts = ["an animal", "a boat", "a human","a building","a buoy"]

file_path="/Users/nicholasabram/TADMUS/Zero Shot Image Segmentation/sea_esc_images"
collection=glob.glob(os.path.join(file_path, '*.jpg'))
count=1

time_collection=[]
time_collect=0
for i in collection:
    image = Image.open(i)

    start_time=time.time()

    inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    end_time=time.time()


    total_time=end_time-start_time
    time_collection.append(total_time)
    time_collect+=total_time

    logits = outputs.logits#prob
    print(logits.shape,'the total time to get the output was',total_time)
    #torch.Size([3, 352, 352])

    import matplotlib.pyplot as plt

    logits = logits.unsqueeze(1)

    _, ax = plt.subplots(1, len(texts) + 1, figsize=(3*(len(texts) + 1), 12))
    [a.axis('off') for a in ax.flatten()]
    #ax[0].imshow(image)

    [ax[i+1].imshow(torch.sigmoid(logits[i][0])) for i in range(len(texts))];
    [ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(texts)]


    ##Insert end of time 
    plt.savefig('output'+str(count)+'.png')
    count+=1

    #my job now is to loop through all these pictures and compare 

average=time_collect/166
standard_deviation=statistics.stdev(time_collection)
q1 = np.percentile(time_collection, 25)
q3 = np.percentile(time_collection, 75)
mini= min(time_collection)
maxi=max(time_collection)
five_percentiles=np.percentile(time_collection,5)
ten_percentiles=np.percentile(time_collection,10)
nine_percentiles=np.percentile(time_collection,90)
ninefive_percentiles=np.percentile(time_collection,95)

print(f'Total data:\n{time_collection}\n\n\n\n\n\n\nThe total time was{time_collect} The average time was:{average}\nThe standard deviation was:{standard_deviation}\nThe q1 was:{q1}\t the q3 was:{q3}\nThe minimum value was:{mini}\nThe maximum value was:{maxi}\nThe 5th percentile was:{five_percentiles}\nThe 10th percentile was:{ten_percentiles}\nThe 90th percentile was:{nine_percentiles}\nThe 95th percentile was:{ninefive_percentiles}')

# """The total time was 151.5101840496 The average time was:0.9127119521060622
# The standard deviation was:0.2204221146525226
# The q1 was:0.8085591793060303	 the q3 was:0.930988073348999
# The minimum value was:0.7362477779388428
# The maximum vale was:2.529625177383423
# The 5th percentile was:0.7822309017181397
# The 10th percentile was:0.7898892402648926
# The 90th percentile was:1.1153455734252928
# The 95th percentile was:1.276258563995361"""      