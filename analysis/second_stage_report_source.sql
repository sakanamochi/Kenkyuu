CREATE VIEW report_model_performance AS
SELECT * FROM source_model_performance ORDER BY parameter_count, evaluation;

CREATE VIEW report_ood_background AS
SELECT * FROM source_ood_background ORDER BY background, method_label;

CREATE VIEW report_ood_tilt AS
SELECT * FROM source_ood_tilt ORDER BY camera_tilt_deg, method_label;

CREATE VIEW report_black_rectangle AS
SELECT * FROM source_black_rectangle ORDER BY severity, method_label;

CREATE VIEW report_cnn_effects AS
SELECT * FROM source_cnn_effects ORDER BY effect_label, severity;

CREATE VIEW report_ransac_selection AS
SELECT * FROM source_ransac_selection ORDER BY success_rate;

CREATE VIEW report_model_latency AS
SELECT * FROM source_model_latency ORDER BY parameter_count, device;

CREATE VIEW report_professor_questions AS
SELECT * FROM source_professor_questions ORDER BY priority;
