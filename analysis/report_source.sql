-- paflab.reporting.build_report_artifact がSQLite上へ投入した監査済み集計を、
-- レポートの表示粒度へ射影・整列する実際のソースクエリ。
CREATE VIEW report_overall AS
SELECT
    method,
    method_label,
    sample_count,
    detected_count,
    success_count,
    success_rate,
    detection_rate,
    success_definition
FROM source_overall
ORDER BY CASE method
    WHEN 'zhang2019_arc_reproduction' THEN 1
    WHEN 'cnn_ransac' THEN 2
    ELSE 99
END;

CREATE VIEW report_curves AS
SELECT
    method,
    method_label,
    degradation,
    degradation_label,
    severity,
    sample_count,
    camera_cluster_count,
    detection_rate,
    success_rate,
    cluster_ci95_low,
    cluster_ci95_high,
    mean_ellipse_iou,
    median_ellipse_iou
FROM source_curves
ORDER BY degradation, severity, method;

CREATE VIEW report_aggregate AS
SELECT
    method,
    method_label,
    degradation,
    degradation_label,
    robustness_auc,
    critical_severity_50
FROM source_aggregate
ORDER BY robustness_auc DESC;
