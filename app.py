import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



books = pd.read_csv('Cleaned Data/cleaned_books.csv')
users = pd.read_csv('Cleaned Data/cleaned_users.csv')
ratings = pd.read_csv('Cleaned Data/cleaned_ratings.csv')


if __name__ == "__main__":
    st.set_page_config(
        page_title="Book Recommendation Application",
        page_icon="📚",
        layout="centered",
    )


    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "Top 50 Books", "Books Popular yearly", "Analysis"],
                               icons=['house', "list-task", "list-task", "gear"],
                               menu_icon="cast", default_index=0)
        selected

        st.write("Bookify: The Ultimate Book Recommendation Application With Data-Driven Intelligence.\n\nThis application will help you to find the best books !!\n\nHappy Reading !!")


    if selected== "Home":
        st.header("Book Recommendation Application")

        model_knn = pickle.load(open('model_knn.pkl', 'rb'))
        books_name = pickle.load(open('books_name.pkl', 'rb'))
        user_rating = pickle.load(open('user_rating.pkl', 'rb'))
        user_rating_pivot2 = pickle.load(open('user_rating_pivot2.pkl', 'rb'))

        selected_books = st.selectbox(
            "Type or select a book",
            books_name
        )


        def fetch_poster(suggestion):
            book_name = []
            ids_index = []
            poster_url = []

            for book_id in suggestion:
                book_name.append(user_rating_pivot2.index[book_id])

            for name in book_name[0]:
                ids = np.where(user_rating['Book-Title'] == name)[0][0]
                ids_index.append(ids)

            for ids in ids_index:
                url = user_rating.iloc[ids]['Image-URL-M']
                poster_url.append(url)
            return poster_url


        def recommend_books(bk_name):
            book_list = []

            # index fetch
            book_id = np.where(user_rating_pivot2.index == bk_name)[0][0]
            distance, suggestion = model_knn.kneighbors(user_rating_pivot2.iloc[book_id, :].values.reshape(1, -1),
                                                        n_neighbors=6)
            poster_url = fetch_poster(suggestion)
            for i in range(len(suggestion)):
                books = user_rating_pivot2.index[suggestion[i]]
                for j in books:
                    book_list.append(j)
                return book_list, poster_url


        if st.button('Show Recommendations'):
            try:
                if selected_books == " ":
                    raise Exception
            except Exception:
                st.text('Please enter book name in the search box! 😢')
            else:
                recommendation_books, poster_url = recommend_books(selected_books)
                for i in range(1, 6):
                    st.text(str(i) + ". " + recommendation_books[i])
                    st.image(poster_url[i])

    if selected == "Top 50 Books":
        st.header("Top 50 books")
        popular_df = pickle.load(open('popular_df.pkl', 'rb'))

        for i in range(len(popular_df)):
            st.text(str(i + 1) + ". " + popular_df.iloc[i][1])
            st.image(popular_df.iloc[i][3])

    if selected ==  "Books Popular yearly":

        st.header("Books popular yearly")
        popular_df_y = pickle.load(open('popular_df_y.pkl', 'rb'))

        for i in range(len(popular_df_y)):
            st.text(str(i + 1) + ". " + " Year: " + str(popular_df_y.iloc[i][3]))
            st.text(popular_df_y.iloc[i][1])

    if selected == "Analysis":
        st.header("Analysis")

        # Graph 1
        fig1=plt.figure(figsize=(10, 8))
        sns.countplot(y="Book-Author", palette='Paired', data=books, order=books['Book-Author'].value_counts().index[0:10])
        plt.title("Analysis 1 - Author with highest no.of books published")

        # Add figure in streamlit app
        st.pyplot(fig1)


        # Graph 2
        fig2=plt.figure(figsize=(10, 8))
        sns.countplot(y="Publisher", palette='Paired', data=books, order=books['Publisher'].value_counts().index[0:10])
        plt.title("Analysis 2 - Top publishers")

        # Add figure in streamlit app
        st.pyplot(fig2)




        explicit_rating = ratings[ratings['Book-Rating'] != 0]

        # Merging  all three datasets
        # for the rating dataset, we are only taking the explicit rating dataset

        df = pd.merge(books, explicit_rating, on='ISBN', how='inner')
        df = pd.merge(df, users, on='User-ID', how='inner')




        # Graph 3

        popular = df.groupby('Book-Title')['Book-Rating'].sum().reset_index().sort_values(by='Book-Rating',
                                                                                          ascending=False)[:10]
        popular.columns = ['Book-Title', 'Count']

        fig3=plt.figure(figsize=[10, 8])
        plt.title('Analysis 3 - Top 10 highest rated books')
        sns.barplot(data=popular, y='Book-Title', x='Count', palette='Set2')

        # Add figure in streamlit app
        st.pyplot(fig3)






        # Graph 4

        author = df.groupby('Book-Author')['Book-Rating'].sum().reset_index().sort_values(by='Book-Rating',ascending=False)[:10]
        fig4=plt.figure(figsize=[10, 8])
        plt.title('Analysis 4 - Top 10 highest rated authors')
        sns.barplot(data=author, y='Book-Author', x='Book-Rating', palette='Set2')

        # Add figure in streamlit app
        st.pyplot(fig4)






        # Graph 5
        year = books['Year-Of-Publication'].value_counts().sort_index()
        year = year.where(year > 5)
        fig5=plt.figure(figsize=(10, 8))
        plt.bar(year.index, year.values)
        plt.xlabel('Year of Publication')
        plt.ylabel('Counts')
        plt.title("Analysis 5 - Number of Books published on yearly basis")

        # Add figure in streamlit app
        st.pyplot(fig5)


        # Graph 6
        fig6=plt.figure(figsize=(10, 8))
        users.Age.hist(bins=[10 * i for i in range(1, 10)], color='cyan')
        plt.title('Analysis 6 - Age distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')


        # Add figure in streamlit app
        st.pyplot(fig6)






        # Graph 7

        fig7=plt.figure(figsize=(10, 8))
        sns.countplot(x="Book-Rating", palette='Paired', data=explicit_rating)
        plt.title('Analysis 7 - Rating distribution')

        # Add figure in streamlit app
        st.pyplot(fig7)






