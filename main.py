import streamlit as st
from ai import Model

st.title("Benchmarking two ML algorithms")

selected_algorithm = st.selectbox("Select the algorithm", ["KNN", "SVC", "Random Forest"])

if selected_algorithm == "KNN":
    st.header("n_neighbors")
    st.text("Number of neighbors to use for kneighbors queries")
    n_neighbors = st.slider("n_neighbors", 1, 50, 5, label_visibility="collapsed")
    st.header("weights")
    st.text("Weight function used in prediction")
    weights = st.selectbox("weights", ["uniform", "distance"], label_visibility="collapsed")
elif selected_algorithm == "SVC":
    st.header("C")
    st.text("Regularization parameter")
    C = st.slider("C", 0.01, 10.0, 1.0, label_visibility="collapsed")
    st.header("kernel")
    st.text("Specifies the kernel type to be used in the algorithm")
    kernel = st.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid"], index = 2, label_visibility="collapsed")
else:
    st.header("n_estimators")
    st.text("Number of trees in the forest")
    n_estimators = st.slider("n_estimators", 1, 500, 100, label_visibility="collapsed")
    st.header("criterion")
    st.text("The function to measure the quality of a split")
    criterion = st.selectbox("criterion", ["gini", "entropy", "log loss"], index=1, label_visibility="collapsed")
    st.header("max_depth")
    st.text("The maximum depth of the tree, use 0 for unlimited depth")
    max_depth = st.number_input("max_depth", 0, 100, 10, label_visibility="collapsed")

if st.button("Run", type="primary"):
    model = Model()
    if selected_algorithm == "KNN":
        model.knn(n_neighbors, weights)
    elif selected_algorithm == "SVC":
        model.svc(C, kernel)
    else:
        if max_depth == -1 or max_depth == 0:
            max_depth = None
        model.random_forest(n_estimators, criterion, max_depth)

    model.fit()
    model.predict()

    st.write("Confusion Matrix", model.confusion_matrix())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", model.accuracy())
    with col2:
        st.metric("Precision", model.precision())
    with col3:
        st.metric("Recall", model.recall())