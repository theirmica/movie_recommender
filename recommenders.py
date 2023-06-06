"""
In this script we define functions for the recommender web
application
"""

def nmf_recommender(query, nmf_model, titles, k=10):
    """This is an nmf-based recommender"""
    
   # def recommend_nmf(query, loaded_model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    # 1. construct new_user-item dataframe given the query
    
    #into dataframe
    new_user_dataframe =  pd.DataFrame(query, columns=titles, index=["new_user"])
    #new_user_dataframe
    
    # using the same imputation as training data

    new_user_dataframe_imputed = new_user_dataframe.fillna(model.mean())
    #new_user_dataframe_imputed
    
    #create user-feature matrix P for new user
    P_new_user_matrix = loaded_model.transform(new_user_dataframe_imputed)
    #P_new_user_matrix

    # get as dataframe for a better visualizarion
    P_new_user = pd.DataFrame(P_new_user_matrix, 
                         columns = loaded_model.get_feature_names_out(),
                         index = ['new_user'])

    #reconstruct the user-movie(item) matrix/dataframe for the new user)
    R_hat_new_user_matrix = np.dot(P_new_user, Q)
    #R_hat_new_user_matrix
    # get as dataframe for a better visualizarion
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=titles,
                         index = ['new_user'])
    #R_hat_new_user
    query.keys()
    
    R_hat_new_user_filtered = R_hat_new_user.drop(query.keys(), axis=1)
    #R_hat_new_user_filtered
    
    R_hat_new_user_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    
    ranked = R_hat_new_user_filtered.T.sort_values(by=["new_user"], ascending=False).index.tolist()
    #ranked
    recommendations = ranked[:10]
   # recommendations
    
    # 2. scoring
    
        # calculate the score with the NMF model
     


    
    # 3. ranking
    
        # filter out movies already seen by the user
        

        # return the top-k highest rated movie ids or titles
  
    return recommendations
    
       
   # return NotImplementedError

def neighbour_recommender(query, cos_sim_model, titles, k=10):
    """This is an cosine-similarity-based recommender"""
    # collaborative filtering = look at ratings only!
#def recommend_neighborhood(query, model, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    #did not do this as for now 
    # construct a user vector
    new_user_dataframe =  pd.DataFrame(query, columns=model.columns, index=['new_user'])
    #new_user_dataframe
    new_user_dataframe_imputed = new_user_dataframe.fillna(Ratings.mean())
    
    # 2. scoring
    
    # find n neighbors
    
    # calculate their average rating
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model2.kneighbors(
                                                        new_user_dataframe_imputed,
                                                        n_neighbors=10,
                                                        return_distance=True
                                                        )

# sklearn returns a list of predictions
# extract the first and only value of the list

    neighbors_df = pd.DataFrame(
                                data = {
                                        'neighbor_id': neighbor_ids[0],
                                        'similarity_score': similarity_scores[0]
                                }
                                )

    #neighbors_df# only look at ratings for users that are similar!
    neighborhood = Ratings.iloc[neighbor_ids[0]]
    #neighborhood
    
    new_user_query.keys()
    
    neighborhood_filtered = neighborhood.drop(new_user_query.keys(), axis=1)
    #neighborhood_filtered
    
    df_score = neighborhood_filtered.sum() #or mean as you prefer
    #df_score
    
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    #df_score_ranked[:10]
    
    recommendations = df_score_ranked[:10]
    
    return recommendations
    
    #return NotImplementedError


