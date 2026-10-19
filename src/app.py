import streamlit as st
from pathlib import Path
from PIL import Image

from search import search


st.set_page_config(
    page_title="WikiArt Image Similarity Search",
    page_icon="🎨",
    layout="wide"
)


def format_artwork_name(image_path):
    """
    Convert the dataset filename into a readable artist and artwork title.
    Example:
    domenico-ghirlandaio_portrait-of-a-young-woman.jpg
    """
    filename = Path(image_path).stem

    if "_" in filename:
        artist, title = filename.split("_", 1)
    else:
        artist = "Unknown Artist"
        title = filename

    artist = artist.replace("-", " ").title()
    title = title.replace("-", " ").title()

    return artist, title


# -----------------------------
# Header
# -----------------------------

st.title("🎨 WikiArt Image Similarity Search")

st.write(
    "Upload an image to find the five most visually similar "
    "artworks from the WikiArt dataset."
)

st.divider()


# -----------------------------
# Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Query image
    st.subheader("Query Image")

    query_column, _ = st.columns([1, 3])

    with query_column:
        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

    # Save temporary query image
    query_path = Path("query_image.jpg")
    image.save(query_path)

    # Search
    with st.spinner("Finding visually similar artworks..."):
        results = search(str(query_path))

    st.divider()

    st.subheader("Top 5 Similar Artworks")

    # Display results
    columns = st.columns(5)

    for rank, (column, result) in enumerate(
        zip(columns, results),
        start=1
    ):

        image_path = Path(result["image"])
        similarity = result["similarity"]

        artist, title = format_artwork_name(image_path)

        with column:

            st.markdown(f"### #{rank}")

            if image_path.exists():
                st.image(
                    str(image_path),
                    width="stretch"
                )

            st.markdown(f"**{title}**")
            st.caption(f"Artist: {artist}")

            st.metric(
                "Similarity",
                f"{similarity:.2%}"
            )


st.divider()

st.caption(
    "Powered by OpenCLIP ViT-B/32 • "
    "Cosine similarity • 10,000 WikiArt images"
)