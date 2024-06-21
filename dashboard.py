import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns



# Load and shuffle the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = shuffle(df, random_state=42)
    return df


# Train data loading 
@st.cache_data
def load_data_df_v1(file_path):
    df_v1 = pd.read_csv(file_path)
    df_v1 = shuffle(df_v1, random_state=42)
    return df_v1


class AIHEALTHGUARDDASHBOARD:
    # Main function to render the dashboard
    @staticmethod
    def main():
        st.title("Welcome to AI-Health Guard Dashboard")
        st.caption("Your Personalized Health Advisor. Predicts diseases, offers tailored medical advice, workouts, and diet plans for holistic well-being.")
        st.subheader("Data Analysis Insight:")
        st.write("Insights from data analysis shed light on trends, patterns and relations between symptoms and health conditions.")
        st.subheader("""
                 """)


        # Define the dataset path
        data_file_path = 'Dataset/Original_Dataset.csv'
        df = load_data(data_file_path)

        # Display the dataset
        if st.checkbox("Show dataset", key="Show df"):
            st.write(df.head())

        # Display data characteristics
        st.subheader("Data Characteristics")
        st.write(df.describe())


        # Frequency of each disease
        st.subheader("Frequency of Each Disease")
        disease_counts = df['Disease'].value_counts()
        fig = px.bar(disease_counts, x=disease_counts.index, y=disease_counts.values, title='', color=disease_counts.index)
        fig.update_layout(xaxis_title='Disease', yaxis_title='Count', xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "Frequency of Each Disease plots represent approx all disease has same count frequency."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        # Word Cloud of Symptoms
        st.subheader("Word Cloud of Symptoms")
        symptom_columns = df.columns[1:]
        all_symptoms = df[symptom_columns].values.flatten()
        all_symptoms = [symptom for symptom in all_symptoms if pd.notna(symptom)]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_symptoms))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        with st.expander("Show Discription"):
            content = "A word shown in `big size` indicates that it is used more times."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        # Frequency Plot of Top Symptoms
        st.subheader("Most Prevalent Symptoms in the Dataset")
        symptom_list = df[symptom_columns].values.flatten()
        symptom_list = [symptom for symptom in symptom_list if pd.notna(symptom)]
        symptom_counts = pd.Series(symptom_list).value_counts().head(20)
        fig = px.bar(symptom_counts, x=symptom_counts.values, y=symptom_counts.index, orientation='h', title='', color=symptom_counts.index)
        fig.update_layout(xaxis_title='Count', yaxis_title='Symptom')
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "Top Reported Symptoms in the Study  is `fatigue`, followed by `vomiting`, `high_fever`, `loss_of_appetite`, and `nausea`, etc..."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        # Symptom Distribution per Disease
        st.subheader("Symptom Distribution per Disease")
        disease = st.selectbox('Select a disease to view symptom distribution', df['Disease'].unique())
        symptom_counts = df[df['Disease'] == disease].iloc[:, 1:].notna().sum()
        fig = px.bar(symptom_counts.sort_values(), orientation='h', labels={'index': 'Symptoms', 'value': 'Count'}, title=f'Symptom Distribution for {disease}', color=symptom_counts.index)
        fig.update_layout(xaxis_title='Count', yaxis_title='Symptoms')
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "The bar chart shows the distribution of symptoms. The x-axis shows the count of people experiencing the symptom, and the y-axis shows the different symptoms. For example, the bar labeled Symptom_1 is the most common symptom, and around 120 people experienced it."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        st.subheader("Disease-Symptom Network Graph")
        st.write("Please make the graph full screen for better view.")
        # Disease-Symptom Network Graph with Interactive Plot
        B = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        diseases = df['Disease'].unique()
        symptoms = pd.melt(df, id_vars=['Disease'], value_vars=df.columns[1:]).dropna()['value'].unique()

        B.add_nodes_from(diseases, bipartite=0)
        B.add_nodes_from(symptoms, bipartite=1)
        # Add edges between diseases and symptoms
        edges = []
        for index, row in df.iterrows():
            for symptom in df.columns[1:]:
                if pd.notna(row[symptom]):
                    edges.append((row['Disease'], row[symptom]))

        B.add_edges_from(edges)
        # Get positions for the nodes in G
        pos = nx.spring_layout(B)
        # Extract the edge and node information
        edge_x = []
        edge_y = []
        for edge in B.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        node_text = []
        for node in B.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center', hoverinfo='text', marker=dict(color=[], size=10, line=dict(width=2)))
        node_trace.marker.color = ['blue' if node in diseases else 'red' for node in B.nodes()]

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(text="Alok Choudhary", showarrow=False, xref="paper", yref="paper")],
                            xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "The network graph shows connections between different diseases and symptoms."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        # Pie Chart of Top Symptoms
        st.subheader("Prevalence of Most Common Symptoms")
        symptom_counts = df.iloc[:, 1:].stack().value_counts().head(10)
        fig = px.pie(symptom_counts, values=symptom_counts.values, names=symptom_counts.index, title='')
        fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "It shows the percentage of people experiencing various symptoms. Fatigue is the most common symptom, affecting 15.8% of people. Other symptoms include high fever, loss of appetite, nausea, headache, abdominal pain, yellowish skin, and vomiting."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        # Parallel Categories Diagram
        st.subheader("Parallel Categories Diagram")
        st.write("Please make the graph full screen for better view.")
        fig = px.parallel_categories(df, color_continuous_scale=px.colors.sequential.Inferno, title="")
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "It show relationships between multiple categorical variables. In this example, the categories are different diseases and symptoms. Each vertical axis represents a single variable, and each data point is a combination of categories across all variables. Lines connect the data points, showing how different categories co-occur."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




        ################################################################################################


        # Visualization on Encoded Dataset
        st.header("Visualization on train Dataset")

        # Define the dataset path
        data_file_path_v1 = 'Dataset/df_v1.csv'
        df_v1 = load_data(data_file_path_v1)

        # Display the dataset
        if st.checkbox("Show dataset", key="Show df_v1"):
            st.write(df_v1.head())



        st.subheader("Exploring Multidimensional Symptom Relationships")
        # Create a multiselect widget to choose symptoms for the 3D scatter plot
        selected_symptoms = st.multiselect("Select Symptoms for Multidimensional Relationships", df_v1.columns)
        # Check if at least three symptoms are selected
        if len(selected_symptoms) >= 3:
            scatter_df = df_v1[selected_symptoms + ['Disease']].dropna()
            fig = px.scatter_3d(scatter_df, x=selected_symptoms[0], y=selected_symptoms[1], z=selected_symptoms[2], 
                                color='Disease', title="")
            st.plotly_chart(fig)
        else:
            st.write("Please select at least three symptoms to generate the 3D Scatter Plot.")
        with st.expander("Show Discription"):
            content = "The plot show the distribution of data points across three dimensions, which represent selected symptoms. Each data point represents a patient, and the position of the point along each axis corresponds to the severity or presence of a particular symptom. For example, one axis might represent muscle weakness, another family history, and the third silver-like dusting. By examining the position of the data points in 3D space, you can see how the different symptoms co-occur in the patient population."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)





        st.subheader("Flows Between Diseases and Symptoms")
        # To show the flow from symptoms to diseases.
        # Define the nodes and links for the Sankey diagram
        nodes = list(df_v1['Disease'].unique()) + list(df_v1.columns[1:10])
        links = []
        for disease in df_v1['Disease'].unique():
            for symptom in df_v1.columns[1:10]:
                count = df_v1[(df_v1['Disease'] == disease) & (df_v1[symptom] == 1)].shape[0]
                if count > 0:
                    links.append({'source': nodes.index(symptom), 'target': nodes.index(disease), 'value': count})
        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes ),
            link=dict(source=[link['source'] for link in links], target=[link['target'] for link in links], value=[link['value'] for link in links])))
        fig.update_layout(title_text="", font_size=10)
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "The diagram show the flow of patients between different diseases and related symptoms. The width of the arrows represents the number of patients experiencing a particular symptom or disease."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)






        st.subheader("Parallel Relationships Between Diseases and Symptoms")
        st.write("Please make the graph full screen for better view.")
        # To visualize the relationships between different symptoms and diseases.
        # Create a parallel categories diagram
        fig = px.parallel_categories(df_v1, dimensions=['Disease', 'family_history', 'muscle_weakness', 'silver_like_dusting'],
                                     title="", color_continuous_scale='viridis')
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "It show relationships between multiple categorical variables. In this example, the categories are different diseases and symptoms. Each vertical axis represents a single variable, and each data point is a combination of categories across all variables. Lines connect the data points, showing how different categories co-occur."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)






        st.subheader("Bubble Chart of Disease Symptom Counts")
        st.write("Please make the graph full screen for better view.")
        # To visualize the occurrence of diseases with varying symptom counts.
        # Create a bubble chart
        df_v1['symptom_count'] = df_v1.iloc[:, 1:].sum(axis=1)
        fig = px.scatter(df_v1, x='Disease', y='symptom_count', size='symptom_count',
                         color='Disease', title='')
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "The chart shows the number of symptoms associated with various diseases. The size of the bubble corresponds to the number of symptoms."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)





        st.subheader("Disease-Symptom Network")
        st.write("Please make the graph full screen for better view.")
        # Create the graph
        G = nx.Graph()
        # Add nodes and edges
        diseases = df_v1['Disease'].unique()
        symptoms = df_v1.columns[1:]
        # Add disease nodes
        for disease in diseases:
            G.add_node(disease, type='disease')
        # Add symptom nodes and edges
        for symptom in symptoms:
            G.add_node(symptom, type='symptom')
            for disease in diseases:
                if df_v1[df_v1['Disease'] == disease][symptom].sum() > 0:
                    G.add_edge(disease, symptom)
        # Define node positions using a spring layout
        pos = nx.spring_layout(G)
        # Create edge traces
        edge_trace = go.Scatter( x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        # Create node traces
        node_trace = go.Scatter( x=[], y=[], text=[], mode='markers+text', textposition='top center', hoverinfo='text',
            marker=dict( showscale=True, colorscale='YlGnBu', size=10, colorbar=dict( thickness=15, title='Node Connections', xanchor='left', titleside='right'), color=[]))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
            node_trace['marker']['color'] += (G.degree(node),)

        # Create the plot
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(text="Network graph of diseases and symptoms (by Alok Choudhary)", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig)
        with st.expander("Show Discription"):
            content = "Nodes in the graph represent various diseases and symptoms. Edges connect nodes that are frequently co-occur together. Thicker edges indicate stronger connections."
            st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)





if __name__ == "__main__":
    AIHEALTHGUARDDASHBOARD.main()
