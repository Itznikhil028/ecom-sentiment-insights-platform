import os
import pandas as pd
import random

os.makedirs("data", exist_ok=True)

products = ["Wireless Headphones", "Smart Fitness Band", "Mechanical Keyboard", "Gaming Mouse"]
reviews_pool = {
    5: [
        "Absolutely amazing! The sound quality is crystal clear and battery lasts forever.",
        "Best purchase of the year! Highly recommend to everyone.",
        "Extremely premium feel, works flawlessly and looks super stylish.",
        "Delivered very fast. The performance is top-notch, totally worth the money!"
    ],
    4: [
        "Pretty good product, quite satisfied with the daily performance.",
        "Value for money. Build quality is solid, but packaging could have been better.",
        "Very nice features and easy to set up. Happy with it so far.",
        "Decent battery backup and comfortable to use for long hours."
    ],
    3: [
        "Average product. It does the job but nothing extraordinary.",
        "Okay for the price, though the plastic build feels a bit cheap.",
        "Delivery took too long. The product itself is decent but not great.",
        "It works fine, but occasionally faces connectivity issues."
    ],
    2: [
        "Not happy with this. The battery drainage is extremely fast.",
        "Disappointed with the quality. It feels very flimsy and fragile.",
        "Stopped working properly after just two days. Bad experience.",
        "The actual product does not match the premium description shown online."
    ],
    1: [
        "Terrible experience! Completely a waste of my hard-earned money.",
        "Horrible quality. The button broke on the very first day of usage.",
        "Extremely poor customer support and cheap materials used.",
        "Worst item ever bought! It arrived in a damaged box and didn't even turn on."
    ]
}

data = []
for i in range(500):
    rating = random.randint(1, 5)
    review = random.choice(reviews_pool[rating])
    product = random.choice(products)
    data.append([product, review, rating])

df = pd.DataFrame(data, columns=["Product_Name", "Review_Text", "Rating"])
df.to_csv('data/reviews.csv', index=False)
print("🎉 Success! data/reviews.csv successfully. !")