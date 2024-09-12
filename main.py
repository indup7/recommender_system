from fastapi import FastAPI, HTTPException
from recommender import Recommender
from utils import load_and_preprocess_data
from typing import List

app = FastAPI()

# Load data and initialize recommender
data, users, products = load_and_preprocess_data()
recommender = Recommender(data["Quantity"], users, products)
recommender.create_and_fit(
    model_name="als",
    model_params=dict(factors=190, alpha=0.6, regularization=0.06, random_state=42),
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommender System API"}
@app.get("/users")
def get_users():
    return {"users": users.values.tolist()}

@app.get("/recommend/{CustomerID}")
def recommend_products(CustomerID: int, n: int = 5):
    print(users.values)
    if CustomerID not in users.values:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        suggestions, _ = recommender.recommend_products(CustomerID, items_to_recommend=n)
        recommended_items = data.loc[data["ProductIndex"].isin(suggestions), "Description"].unique()
        return {"CustomerID": CustomerID, "recommended_items": recommended_items.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar-users/{CustomerID}")
def similar_users(CustomerID: int):
    if CustomerID not in users.values:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        sim_users, _ = recommender.similar_users(CustomerID)
        return {"CustomerID": CustomerID, "similar_users": sim_users.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{CustomerID}/{product_id}")
def explain_recommendation(CustomerID: int, product_id: int):
    if CustomerID not in users.values:
        raise HTTPException(status_code=404, detail="User not found")
    if product_id not in products.values:
        raise HTTPException(status_code=404, detail="Product not found")
    
    try:
        explanation = recommender.explain_recommendation(CustomerID, product_id, recommended_items=5)
        return {"CustomerID": CustomerID, "product_id": product_id, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the application, use the following command:
# uvicorn main:app --reload
