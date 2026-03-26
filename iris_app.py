import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score 

st.set_page_config(page_title="Iris EDA and ML", page_icon="🌸", layout="wide")
page = st.sidebar.radio("Go to " , [
    "1- Overview" , 
    "2- Overlap" , 
    "3- Feature Importance" , 
    "4- Machine Learning" , 
    "5-Predictor"
]) 
st.sidebar.markdown("---") 
st.sidebar.markdown("""
**Key Findings:**
- Setosa → 100% separable
- Versicolor vs Virginica → real overlap
- Real ceiling → ~97%
""")

data = pd.read_csv('Iris.csv')
data = data.drop('Id', axis=1)

setosa     = data[data['Species'] == 'Iris-setosa']
versicolor = data[data['Species'] == 'Iris-versicolor']
virginica  = data[data['Species'] == 'Iris-virginica']

if page == "1- Overview" : 
    st.title("Iris Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", 150)
    col2.metric("Features", 4)
    col3.metric("Species", 3)
    col4.metric("Missing Values", 0)

    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(data.head(10))

    st.subheader("Data Summary")
    st.dataframe(data.describe().round(2))

    st.markdown("---")
    st.title("Iris Overlap")

    feature = st.selectbox("Select Feature",
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(setosa[feature],     bins=15, alpha=0.6, color='#3a7bd5', label='Setosa')
        ax.hist(versicolor[feature], bins=15, alpha=0.6, color='#e8a020', label='Versicolor')
        ax.hist(virginica[feature],  bins=15, alpha=0.6, color='#c0392b', label='Virginica')
        ax.set_xlabel(feature)
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Boxplot")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [setosa[feature], versicolor[feature], virginica[feature]],
            labels=['Setosa', 'Versicolor', 'Virginica']
        )
        st.pyplot(fig)
        plt.close()

elif page =="2- Overlap" :
    st.markdown("---")
    st.subheader("Overlap in PetalLength")

    v_max  = versicolor['PetalLengthCm'].max()
    vg_min = virginica['PetalLengthCm'].min()

    overlap_ver = versicolor[versicolor['PetalLengthCm'] >= vg_min]
    overlap_vg  = virginica[virginica['PetalLengthCm']   <= v_max]

    col1, col2, col3 = st.columns(3)
    col1.metric("Overlap zone",         f"{vg_min:.2f} – {v_max:.2f} cm")
    col2.metric("Versicolor Overlap",   f"{len(overlap_ver)} / 50")
    col3.metric("Virginica Overlap",    f"{len(overlap_vg)} / 50")

    st.warning("PetalLength has overlap between Versicolor and Virginica — adding PetalWidth reduces it from 23 to 4 plants.")

    st.markdown("---")
    st.subheader("Versicolor vs Virginica")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(setosa['PetalLengthCm'],     setosa['PetalWidthCm'],     color='#3a7bd5', label='Setosa',     alpha=0.7, s=40)
    ax.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], color='#e8a020', label='Versicolor', alpha=0.7, s=40)
    ax.scatter(virginica['PetalLengthCm'],  virginica['PetalWidthCm'],  color='#c0392b', label='Virginica',  alpha=0.7, s=40)
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_title('All Species')
    ax.legend()

    ax = axes[1]
    ax.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], color='#e8a020', label='Versicolor', alpha=0.7, s=40)
    ax.scatter(virginica['PetalLengthCm'],  virginica['PetalWidthCm'],  color='#c0392b', label='Virginica',  alpha=0.7, s=40)
    ax.axvspan(vg_min, v_max, color='gray', alpha=0.15, label='Overlap Zone')
    ax.axvline(vg_min, color='gray', linestyle='--')
    ax.axvline(v_max,  color='gray', linestyle='--')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_title('Zoomed Overlap')
    ax.legend()

    st.pyplot(fig)
    plt.close() 

elif page == "3- Feature Importance" :
    st.markdown("---") 
    st.title("What is most important Feature?") 
    st.subheader("Pairplot") 
    fig = sns.pairplot(data , hue= 'Species' , diag_kind='kde', palette=['#3a7bd5', '#e8a020', '#c0392b']) 
    st.pyplot(fig) 
    plt.close() 
    st.markdown("---") 

    st.subheader("Correlation Heatmap") 
    col1 , col2 = st.columns(2) 

    with col1 : 
        st.write("Correlation Heatmap for all features") 
        fig = plt.figure(figsize=(10,8)) 
        ax = fig.add_subplot(1,1,1)
        sns.heatmap(data.drop('Species' , axis =1 ).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig) 
        plt.close() 

    with col2 : 
        st.write("Correlation Heatmap for Per Species") 
        fig = plt.figure(figsize=(18,4)) 
        ax = fig.add_subplot(1,3,1)
        species = st.selectbox("Select Species", ['Setosa', 'Versicolor', 'Virginica'])
        data_sp = data[data['Species'] == f'Iris-{species}'] 
        sns.heatmap(data_sp.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm', ax = ax ) 
        st.pyplot(fig) 
        plt.close() 
        st.warning("The Total correlation says 0.96 — but when you calculate it for each type you get a different number. This is Simpson's Paradox.") 
        plt.close() 

    st.subheader("Feature Importance (EDA)") 
    importance = pd.DataFrame({ 
        'Feature': ['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm'],

        'Chapter Strength': ['Very Strong', 'Strong', 'Weak', 'Very Weak'],

        'Overlap': ['Small', 'Low Overlap', 'Very Large', 'No Overlap'],

        'Importance': ['★★★★★', '★★★★', '★★', '★']
    }) 
    st.table(importance)
    st.info("PetalLength and PetalWidth are the most important features for distinguishing between the species, while Sepal features are less informative.")

elif page == "4- Machine Learning" :
    st.markdown("---")
    st.title("Machine Learning Models")

    col1, col2 = st.columns(2)
    test_size    = col1.slider("Test Size",     0.1, 0.5, 0.2, step=0.05)
    random_state = col2.slider("Random State",  0, 100, 42,  step=1)
    cv_folds     = st.slider("CV Folds",        2, 10,  5,   step=1)

    X = data.drop('Species', axis=1)
    le = LabelEncoder()
    y  = le.fit_transform(data['Species'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN':                 KNeighborsClassifier(),
        'Decision Tree':       DecisionTreeClassifier(),
        'SVM':                 SVC()
    }

    results = {}

    with st.spinner("Training models..."):
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc    = accuracy_score(y_test, y_pred)
            results[name] = {
                'CV Score':      scores.mean(),
                'cv_std':        scores.std(),
                'scores':        scores,
                'Test Accuracy': acc,
                'Y pred':        y_pred
            }

    st.markdown("---")
    st.subheader("CV Accuracy")
    col1, col2, col3, col4 = st.columns(4)
    for col, (name, res) in zip([col1, col2, col3, col4], results.items()):
        col.metric(name, f"{res['CV Score']:.2%}", f"± {res['cv_std']:.2%}")

    st.markdown("---")
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, res) in zip(axes.flatten(), results.items()):
        cm = confusion_matrix(y_test, res['Y pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f"{name} — Acc: {res['Test Accuracy']:.2%}", fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("CV Scores Table")
    cv_df = pd.DataFrame(
        {name: res['scores'].round(3) for name, res in results.items()}
    )
    cv_df.index = [f"Fold {i+1}" for i in range(cv_folds)]
    st.dataframe(cv_df, use_container_width=True)

    st.warning("If a fold scores lower than others — that is not a model weakness. That is the overlap zone showing up.")

elif page == "5-Predictor" : 
    st.title("Predictor")
    st.info("This is a simple predictor based on the best model (SVM) to predict the species of iris based on user input features.") 

    x = data.drop('Species' , axis = 1 ) 
    le = LabelEncoder() 
    y = le.fit_transform(data['Species']) 
    scaler = StandardScaler() 
    x_scaled = scaler.fit_transform(x) 

    trained_models = { 
        'Logistic Regression': LogisticRegression(max_iter=1000).fit(x_scaled, y), 
        'KNN' :                KNeighborsClassifier().fit(x_scaled, y), 
        'Decision Tree':       DecisionTreeClassifier().fit(x_scaled, y), 
        'SVM':                 SVC(probability=True).fit(x_scaled, y) 

    }

    col1, col2 = st.columns(2) 
    with col1:
        st.subheader("Measurements")
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
        pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)
        model_choice = st.selectbox("Model", list(trained_models.keys()))
    
    with col2 : 
        st.subheader("Result") 
        sample = scaler.transform([[sl, sw, pl, pw]]) 
        model = trained_models[model_choice] 
        pred = model.predict(sample)[0] 
        species = le.classes_[pred] 

        color_map = {
            'Iris-setosa':     '#3a7bd5',
            'Iris-versicolor': '#e8a020',
            'Iris-virginica':  '#c0392b'
        }
        color = color_map[species]

        st.markdown(f"""
        <div style="padding:1.5rem;border-radius:8px;border:2px solid {color};">
            <div style="font-size:0.8rem;color:#888;">Predicted Species</div>
            <div style="font-size:2rem;font-weight:600;color:{color}">{species}</div>
        </div>
        """, unsafe_allow_html=True)

        probs = model.predict_proba(sample)[0] 
        prob_df = pd.DataFrame({
            'Species': le.classes_,
            'Probability': probs
        }).sort_values('Probability', ascending=False)
        st.markdown("---")
        st.bar_chart(prob_df.set_index('Species'))

        st.markdown("**Why this prediction?**")
        if pl < 2.5:
            st.success(f"Petal Length {pl}cm < 2.5 → definitely Setosa. No other species reaches this range.")
        elif pl > 5.1:
            st.info(f"Petal Length {pl}cm > 5.1 → above Versicolor's max. Very likely Virginica.")
        elif pl < 4.5:
            st.info(f"Petal Length {pl}cm is in Versicolor's main range (3.0–4.5). Likely Versicolor.")
        else:
            st.warning(f"Petal Length {pl}cm is in the overlap zone (4.5–5.1). Model is using Petal Width {pw}cm to decide.")

    st.markdown("---")
    st.subheader("Your input in the data")
    fig, ax = plt.subplots(figsize=(8, 5))
    for sp, color in color_map.items():
        df_sp = data[data['Species'] == sp]
        ax.scatter(df_sp['PetalLengthCm'], df_sp['PetalWidthCm'],
                color=color, label=sp, alpha=0.5, s=40)
    ax.scatter(pl, pw, color='black', s=200, zorder=5, marker='*', label='Your input')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig) 
    plt.close()
