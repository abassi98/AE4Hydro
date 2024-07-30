library(intRinsic)
# path = file.path(getwd(), "PHD", "Attention4Hydro", "analysis", "stats", "attributes.csv")
for (seed in 300:303) {
  # select proper number of features
  path = file.path(getwd(), "encoded", paste(paste("encoded_enca_2_", seed, sep=""),".csv", sep=""))
  encoded_features_lstm_ae <- read.csv(path, sep=" ")
  encoded_features_lstm_ae
  basin_ids = encoded_features_lstm_ae[,1]
  basin_ids
  data =encoded_features_lstm_ae[,2:ncol(encoded_features_lstm_ae)]
  
  # print stats
  mean = colMeans(data)
  std = apply(data, 2, sd)
  print(mean)
  print(std)
  
  gride = gride_evolution(data, vec_n1 = 1:280, vec_n2 = 2*(1:280), upp_bound=100 )
  
  id <- gride$path
  d <- gride$avg_distance_n2
  # save id gride
  save_path = file.path(getwd(), "encoded", paste(paste("id_enca_2_", seed, sep=""),".txt", sep=""))
  df_save <- data.frame(d = d, id = id)
  write.csv(df_save, save_path, row.names = FALSE)
}
  