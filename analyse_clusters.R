library(intRinsic)
library(glue)
library(jsonlite)

# arguments
encoded_features = 5
seed = 300
run = 27
nsim = 20000
burn_in = 10000
K = 3
alpha_Dirichlet = 1.0

path = file.path(getwd(), glue("encoded_global_lstm_ae_{encoded_features}_{seed}.txt"))
print(path)
encoded_features_lstm_ae <- read.csv(path, sep=" ")
basin_ids = encoded_features_lstm_ae[,1]
X = encoded_features_lstm_ae[,2:ncol(encoded_features_lstm_ae)]

# create subfolder to save the run
dir.create(file.path(getwd(),  "hidalgo", glue("run{run}")));

# run gride on the full space
gride_full = gride_evolution(X, vec_n1 = 1:283, vec_n2 = 2*(1:283) )
ap_gride_full = autoplot(gride_full)
png(filename=file.path(getwd(),  "hidalgo", glue("run{run}"), glue("gride_full_enca_{encoded_features}_{seed}.png")))
plot(ap_gride_full)
dev.off()



# hidalgo
hid = Hidalgo(X, nsim=nsim, burn_in=burn_in, K=K,alpha_Dirichlet=alpha_Dirichlet)
ap_hid = autoplot(hid)
png(filename=file.path(getwd(), "hidalgo", glue("run{run}"), "hid_autoplot.png"))
plot(ap_hid)
dev.off()


# clustering
cl= clustering(hid, clustering_method = "salso")
print(cl)
label =cl$clust
lab =unique(label)
n_clust = length(lab)

for (c in 1:n_clust){
  X_c = X[which(label==lab[c]),]
  n1 = floor(dim(X_c)[1] / 2.0)-1
  gride_c = gride_evolution(X_c, vec_n1 = 1:n1, vec_n2 = 2*(1:n1) )
  ap_gride = autoplot(gride_c)
  png(filename=file.path(getwd(),  "hidalgo", glue("run{run}"), glue("gride_c{c}_enca_{encoded_features}_{seed}.png")))
  plot(ap_gride)
  dev.off()
  basins = basin_ids[which(label==lab[c])] 
  path = file.path(getwd(), "hidalgo", glue("run{run}"), glue("c{c}_enca_{encoded_features}_{seed}.txt"))
  write.csv(basins, file = path)
}
# save hyper-parameters used
params <- list(encoded_features=27,
                 seed=300, 
                 nsim=nsim,
                 burn_in=burn_in,
                 K=K,
                 alpha_Dirichlet=alpha_Dirichlet)

path = file.path(getwd(), "hidalgo", glue("run{run}"), glue("cfg.json"))
jsonlite::write_json(params, path)

# run python script
py_script <- glue("analyse_clusters.py --run={run} --seed={seed} --encoded_features={encoded_features}")
system(paste("python", py_script))