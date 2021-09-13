# Generic examples of each autonomy pipeline. Intended for demonstration and validating basic installation.

echo "Running the OWLS Autonomy Demo..."

echo "Downloading necessary data..."
curl https://ml.jpl.nasa.gov/projects/owls/owls_demo_data.zip -o owls_demo_data.zip
unzip owls_demo_data.zip
rm owls_demo_data.zip

echo "Running the HELM Autonomy demo pipeline..."
HELM_pipeline --experiments "owls_demo_data/HELM/*" --steps preproc validate tracker features predict asdp manifest --batch_outdir logs

echo "Running the FAME Autonomy demo pipeline..."
FAME_pipeline --experiments "owls_demo_data/FAME/*" --steps preproc validate tracker features predict asdp manifest --batch_outdir logs

echo "Running the ACME Autonomy demo pipeline..."
ACME_pipeline --data "owls_demo_data/ACME/*"

echo "Running the HIRAILS Autonomy demo pipeline..."
HIRAILS_pipeline --experiments "owls_demo_data/HIRAILS/*" --steps tracker asdp --batch_outdir logs

echo "Running the JEWEL Autonomy demo pipeline..."
update_asdp_db "owls_demo_data/HELM/2021_02_03_dhm_yes_low_chlamy_xx_xx_grayscale_lab_16/asdp/" "owls_demo_data/JEWEL/asdp_db.csv"
JEWEL "owls_demo_data/JEWEL/asdp_db.csv" "owls_demo_data/JEWEL/JEWEL_priority.csv"