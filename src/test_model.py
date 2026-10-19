from model import get_embedding

image_path = "test.jpg"

embedding = get_embedding(image_path)

print("Embedding generated successfully")
print("Shape:", embedding.shape)
print("First 5 values:", embedding[:5])