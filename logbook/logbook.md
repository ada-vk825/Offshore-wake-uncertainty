# Logbook

## 27/05/2026 – 13:00

Met with Simon and Matt for the initial IRP discussion along with other students. Simon provided an overview and context of the IRP project. During the discussion, I introduced my idea of incorporating a machine learning component to quantify uncertainty in wake loss and energy yield loss predictions.

### Next Steps
- Review papers related to ML-based wake modelling.
- Investigate surrogate modelling approaches for computational efficiency.
- Explore uncertainty quantification methods for wind farm yield prediction.

# Meeting (28 May 2026)

**Present:** Varun Kokkiligadda (student), Simon Warder (SW, supervisor)

## Key points discussed:

- Discussed the proposed IRP workflow for inter-farm wake modelling and uncertainty quantification using machine learning surrogate models.

- Simon suggested selecting a realistic target offshore wind farm with neighbouring lease areas nearby in order to create plausible inter-farm wake interaction scenarios.

- Discussed using EMODnet offshore wind lease-area datasets instead of relying only on 4C Offshore maps, since EMODnet provides downloadable geospatial data suitable for analysis in Python and QGIS.

- Discussed generating synthetic turbine layouts within realistic lease-area boundaries to quantify uncertainty with respect to neighbouring farm design assumptions such as turbine spacing and layout density.

- Simon recommended breaking the project into intermediate stages, beginning with single-farm wake modelling before extending to multi-farm inter-farm wake scenarios.

- Discussed using PyWake with the TurbOPark model for wake simulations and using ERA5 wind rose data as atmospheric input.

## Feedback received:

- Simon advised reproducing a simpler single-farm case initially before attempting large-scale multi-farm uncertainty studies.

- Simon suggested generating a training dataset for the single-farm case first while continuing to expand the dataset for neighbouring multi-farm scenarios in parallel.

- Simon recommended using realistic neighbouring lease areas and varying layout parameters rather than fully random farm placement.

- Simon suggested exploring the EMODnet dataset using QGIS and geopandas for easier inspection and processing of geospatial lease-area data.

## Work plan before next meeting:

- Download and inspect EMODnet offshore wind lease-area datasets using QGIS and geopandas.

- Select a target case-study offshore wind farm with nearby neighbouring lease areas.

- Set up the initial PyWake + TurbOPark simulation workflow for a single-farm case.

## 04/06/2026 – 13:00

**Attendees:** Varun, Simon, Eirini, Yuzin, and Yan

### Discussion
- Attended a project progress meeting with Simon and fellow IRP students.
- Each student provided an update on their current project status and progress.
- Discussed the literature reviewed so far and how it relates to the respective research topics.
- Reviewed the progress made on the Project Plan, including the proposed methodology, objectives, and research scope.
- I presented my progress on literature related to offshore wind farm wake modelling, machine learning surrogate models, and uncertainty quantification.

### Feedback and Guidance
- Simon provided feedback on the project direction and expectations for the Project Plan.
- Simon suggested to send him the Project Plans by Monday so that he can review them and provide feedback.

### Action Items
- Finalise and submit the Project Plan by Monday.


## 16/06/2026 – 16:00

**Attendees:** Varun, Simon, Eirini, Matt

### Discussion
- Attended a project progress meeting with Simon, Matt and fellow IRP students.
- Gave an update on my work. (Downloaded the ERA5 data, created site objects and turbine objects, created a function to randomly place turbines in the site.)

### Future Plan
- Start simulating and generate the training data
- Finish generating the data before the next meeting.


## 22/06/2026 – Meeting with Simon

**Attendees:** Varun, Simon

### Discussion
- Met with Simon to discuss my proposed approach for developing the machine learning model.
- I asked whether I should replicate the exact input features used in the literature or use a simpler set of features, including:
  - Neighbour wind farm specifications
  - Target wind farm specifications
  - Wind rose data
- My concern was whether these features would be sufficient instead of using the geometric features proposed in the literature.

### Feedback and Guidance
- Simon advised me to follow the approach presented in the literature because the long-term objective of the project is to develop a model that can generalise well to different wind farm layouts.
- He recommended implementing the geometric features used in the published work, as these better capture the spatial relationships between wind farms.
- However, he also clarified that it is **not necessary to implement every geometric feature** mentioned in the paper. Only the features that are most relevant and practical for the project need to be included.

### Action Items
- Investigate how to extract and implement the geometric features described in the literature.
- Identify the subset of geometric features that are most suitable for the project.
- Continue designing the machine learning dataset using the literature as the primary reference for feature engineering.


## 06/07/2026 – Meeting with Simon

**Attendees:** Varun, Simon, Matt, Eirini, Arushi, Chen Yuixn

### Discussion

- Met with Simon and Matt to update them on the progress
- Presented them the code and the initial results of the model.
- Updated them with plan over the next one or two weeks.

### Feedback

- Simon suggested benchmarking my model with the existing literature.
- Matt suggested that sometimes we don't know why the results are the way they are and recommended to understand it. This is what makes the IRP strong.
## 21/07/2026 – Progress Meeting with Simon

**Attendees:** Varun, Simon

### Project Progress
- Implemented a function to run Monte Carlo simulations across all generated scenarios.
- Extracted summary statistics from the simulations to quantify uncertainty in inter-farm wake loss predictions.
- Benchmarked the simulation results against the topology-aware surrogate model paper. The overall trends were consistent with the findings reported in the literature, providing confidence in the implementation.
- Performed a sensitivity analysis by varying:
  - Neighbour farm distance
  - Wind speed
  - Wind direction
- The results showed physically meaningful behaviour:
  - Wake loss decreased as the neighbour farm distance increased.
  - The model captured sensible variations in wake loss for different wind speeds and wind directions.

### Discussion
- Presented the sensitivity analysis results to Simon.
- One observation was that the machine learning model produced **negative R² scores** for some high wind speed scenarios (around the turbine rated wind speed).
- I explained that I had investigated this issue and found that a negative R² score does not necessarily indicate poor model performance. At high wind speeds, the wake loss exhibits very little variation because the turbine is operating near its rated power, making R² less informative.
- Simon agreed that this explanation was reasonable but suggested validating the results using **Root Mean Squared Error (RMSE)**, as RMSE would provide a clearer indication of the prediction error when the target values have low variance.

### Feedback and Guidance
- Use RMSE alongside R² when evaluating model performance, particularly for scenarios with limited variance in the target variable.
- Avoid relying solely on R² for interpreting model accuracy in high wind speed (near-rated speed) conditions.

### Action Items
- Calculate and analyse the RMSE for the high wind speed scenarios.
- Compare both RMSE and R² to obtain a more complete assessment of model performance.
- Continue developing the Streamlit application for demonstrating the uncertainty quantification workflow.


## 18/08/2026 – Progress Meeting with Simon

**Attendees:** Varun, Simon, Eirini

### Project Progress
- Finished packaging the code.
- Wrote unit test cases for the package files.

### Action Items
- Finish the first draft of the report and submit it to Simon by thursday morning.