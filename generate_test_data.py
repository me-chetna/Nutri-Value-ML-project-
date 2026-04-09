import pandas as pd
import random

rows = []

ingredients_pool = [
    "oats, almonds, milk",
    "wheat flour, sugar, palm oil",
    "refined sugar, artificial color",
    "chicken, olive oil, spices",
    "corn syrup, sodium benzoate",
    "vegetables, beans, spices",
    "trans fat, monosodium glutamate",
    "fruit concentrate, aspartame",
    "milk, cocoa, sugar",
    "brown rice, nuts, seeds"
]

for i in range(120):  # 120 rows
    sugar = random.randint(0, 30)
    protein = random.randint(1, 25)
    fat = random.randint(1, 25)
    sodium = random.randint(50, 1000)
    carbs = random.randint(10, 70)
    fiber = random.randint(0, 10)
    calories = (4 * protein) + (4 * carbs) + (9 * fat)

    ingredients = random.choice(ingredients_pool)

    rows.append([
        calories, sugar, protein, fat, sodium, carbs, fiber, ingredients
    ])

df = pd.DataFrame(rows, columns=[
    "calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber", "ingredients"
])

df.to_csv("test_data.csv", index=False)

print("✅ test_data.csv created successfully!")