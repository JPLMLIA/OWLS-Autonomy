# Generic examples of each autonomy pipeline. Intended for demonstration and validating basic installation.

echo "Running the OWLS Autonomy Demo..."

echo "Downloading necessary data..."
curl https://ml.jpl.nasa.gov/projects/owls/owls_demo_data.zip -o owls_demo_data.zip
unzip owls_demo_data.zip
rm owls_demo_data.zip

echo "Running the HELM Autonomy demo pipeline..."
HELM_flight_pipeline --experiments "owls_demo_data/HELM/*" --steps preproc validate tracker features predict asdp manifest --batch_outdir logs
HELM_ground_pipeline --experiments "owls_demo_data/HELM/*" --steps validate asdp manifest --batch_outdir logs

echo "Running the FAME Autonomy demo pipeline..."
FAME_flight_pipeline --experiments "owls_demo_data/FAME/*" --steps preproc validate tracker features predict asdp manifest --batch_outdir logs
FAME_ground_pipeline --experiments "owls_demo_data/FAME/*" --steps validate asdp manifest --batch_outdir logs

echo "Running the ACME Autonomy demo pipeline..."
ACME_flight_pipeline --data "owls_demo_data/ACME/*.pickle"

echo "Running the HIRAILS Autonomy demo pipeline..."
HIRAILS_flight_pipeline --experiments "owls_demo_data/HIRAILS/*" --steps tracker asdp --batch_outdir logs

echo "Running the CSM Autonomy demo pipeline..."
CSM_flight_pipeline "owls_demo_data/CSM/10-Dil-5.csv"

echo "Running the JEWEL Autonomy demo pipeline..."
update_asdp_db --rootdirs "owls_demo_data/HELM/2021_02_03_dhm_yes_low_chlamy_xx_xx_grayscale_lab_16/asdp/" --dbfile "owls_demo_data/JEWEL/asdp_db.csv"
JEWEL --dbfile "owls_demo_data/JEWEL/asdp_db.csv" --outputfile "owls_demo_data/JEWEL/JEWEL_priority.csv"