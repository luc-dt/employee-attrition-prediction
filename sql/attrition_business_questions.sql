-- SQL examples for translating stakeholder retention questions into analysis-ready metrics.
-- Assumes a table named hr_employee_attrition with columns matching HR_Employee_Attrition.csv.

-- 1) Attrition rate by overtime status: useful for workload and staffing decisions.
SELECT
    OverTime,
    COUNT(*) AS employee_count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
    ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM hr_employee_attrition
GROUP BY OverTime
ORDER BY attrition_rate_pct DESC;

-- 2) Early-tenure attrition by department: helps prioritize onboarding interventions.
SELECT
    Department,
    CASE
        WHEN YearsAtCompany < 1 THEN '<1 year'
        WHEN YearsAtCompany BETWEEN 1 AND 2 THEN '1-2 years'
        WHEN YearsAtCompany BETWEEN 3 AND 5 THEN '3-5 years'
        ELSE '6+ years'
    END AS tenure_band,
    COUNT(*) AS employee_count,
    ROUND(100.0 * AVG(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END), 2) AS attrition_rate_pct
FROM hr_employee_attrition
GROUP BY Department, tenure_band
ORDER BY attrition_rate_pct DESC, employee_count DESC;

-- 3) Business-travel and commute risk segment for retention campaign targeting.
SELECT
    BusinessTravel,
    CASE
        WHEN DistanceFromHome <= 5 THEN '0-5 miles'
        WHEN DistanceFromHome <= 15 THEN '6-15 miles'
        ELSE '16+ miles'
    END AS commute_band,
    COUNT(*) AS employee_count,
    ROUND(AVG(JobSatisfaction), 2) AS avg_job_satisfaction,
    ROUND(100.0 * AVG(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END), 2) AS attrition_rate_pct
FROM hr_employee_attrition
GROUP BY BusinessTravel, commute_band
HAVING COUNT(*) >= 10
ORDER BY attrition_rate_pct DESC;
