#!/bin/bash
# Store the Train Version
RESULTS_DIR="$(./params_parser train.save_path)/train"
version="$(./params_parser version)"
pipelineUrl="$CI_PROJECT_URL/-/pipelines/$CI_PIPELINE_ID"
TEST_DIR="$(./params_parser train.save_path)/test"
# Store the model version and weights as URL
echo "$version; $pipelineUrl; $CI_COMMIT_SHORT_SHA" > "$(./params_parser train.save_path)/train/model.version"
echo "Hola"
echo "${RESULTS_DIR}/weights/best.pt($(cat $RESULTS_DIR/weights/best.pt | jisap-cli publish --url))"
#$(cat $RESULTS_DIR/weights/best.pt | jisap-cli publish --url)>"${RESULTS_DIR}/model.pt.url"
echo "[Trained weights: ${RESULTS_DIR}/weights/best.pt]($(cat $RESULTS_DIR/weights/best.pt | jisap-cli publish --url|tee ${RESULTS_DIR}/model.pt.url ))"
cat "${RESULTS_DIR}/model.pt.url"

python3 src/train/save_best_metrics.py -m "$RESULTS_DIR/weights/best.pt" -r "$RESULTS_DIR/results.csv" -s "$(./params_parser train.save_path)/metrics"

(
[ -z "$BRANCH" ] && echo "# Training Report" || echo "# Training Report ($CI_DEFAULT_BRANCH vs $BRANCH)"

echo "## Commit Message"
echo
echo "$CI_COMMIT_MESSAGE" | awk 1 ORS='\n\n'
echo
echo

echo "## Validation Metrics"
echo
dvc metrics diff --all --md || echo "(error performing dvc metrics diff command)"

echo 
echo "## Trained Models "
echo

V=$(cat $RESULTS_DIR/model.version)
echo "[Model version:   "${V%%;*}" ]($(cat $RESULTS_DIR/model.version | jisap-cli publish --url))"
echo
echo "[Best weights: ${RESULTS_DIR}/weights/best.pt]($(cat $RESULTS_DIR/weights/best.pt | jisap-cli publish --url))"
echo
echo "[Last epoch weights: ${RESULTS_DIR}/weights/last.pt ]($(cat $RESULTS_DIR/weights/last.pt | jisap-cli publish --url))"
echo

echo "### Test Metrics Plots"
echo
for f in $(find ${TEST_DIR} | grep png)
do
    echo
    echo "#### $f"
    jisap-cli publish --content-type "image/png" "$f"
    echo
done 
echo
echo "### Test Data Visualization"
echo
for f in $(find ${TEST_DIR} | grep jpg)
do
    echo
    echo "#### $f"
    jisap-cli publish --content-type "image/jpg" "$f"
    echo
done 
echo

echo "### Training Summary"

echo "#### Training Labels Overview$f"
jisap-cli publish --content-type "image/jpg" "${RESULTS_DIR}/labels.jpg"
echo

for f in $(find ${RESULTS_DIR} | grep png)
do
    echo
    echo "#### Training Metrics $f"
    jisap-cli publish --content-type "image/png" "$f"
    echo
done

echo "## Training Accuracy Progress"
echo
dvc plots show -x epoch -y loss --title LOSS --show-vega  ${RESULTS_DIR}/results.csv | vl2svg | jisap-cli publish --content-type image/svg+xml


echo "## Training Logs"
echo
echo "### Training Parameters"
echo
dvc params diff --all --md || echo "(error performing dvc params diff command)"
echo
echo
echo "### Pipeline Info"
echo
echo
echo "* Triggered: $(date -d "$CI_COMMIT_TIMESTAMP" +%c) by $(echo $CI_COMMIT_AUTHOR | awk -F ' <|>' '{printf "[%s](mailto:%s)",$1,$2}')"
echo "* Execution finished: $(date +%c)"
echo "* CI/CD Logs: [link]($CI_JOB_URL)"
echo
echo "###  Versions overview"
echo
src/versions.sh
echo

echo "# Previous Training Results"

jisap-cli dvc-metrics-log --project-ids $CI_PROJECT_ID --pageLen 1 --mdDateFormat "Mon Jan 2 2006 15:04:05 MST" --mdShortIdUrl --mdStagePrefix / --mdMetricsFilePrefix / --mdNoTitle --mdAddGitlabColumns  || echo "(error performing jisap-cli dvc-metrics-log command)"
) > report.md
