/**
 * Hospital Appointment No-Show Decision-Support System - Frontend Client
 * Author: Rohit Singh Negi (Reg No: 12311143)
 */

document.addEventListener('DOMContentLoaded', () => {
  initializeDates();
});

// Format date helpers
function formatDate(d) {
  return d.toISOString().split('T')[0];
}

function formatDateTimeLocal(d) {
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function initializeDates() {
  const now = new Date();
  const schedInput = document.getElementById('scheduled_day');
  const apptInput = document.getElementById('appointment_day');

  if (schedInput && !schedInput.value) {
    schedInput.value = formatDateTimeLocal(now);
  }

  if (apptInput && !apptInput.value) {
    const futureDate = new Date();
    futureDate.setDate(now.getDate() + 14);
    apptInput.value = formatDate(futureDate);
  }
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

  event.currentTarget.classList.add('active');
  const target = document.getElementById(tabId);
  if (target) {
    target.classList.add('active');
  }
}

// Preset Loader
function loadPreset(presetType) {
  const now = new Date();
  const schedInput = document.getElementById('scheduled_day');
  const apptInput = document.getElementById('appointment_day');

  schedInput.value = formatDateTimeLocal(now);

  if (presetType === 'high_risk') {
    // 28 days lead time, 24 years old, past no-show, long distance, no SMS
    const apptDate = new Date();
    apptDate.setDate(now.getDate() + 28);
    apptInput.value = formatDate(apptDate);

    document.getElementById('patient_id').value = '108842';
    document.getElementById('age').value = '23';
    document.getElementById('gender').value = 'M';
    document.getElementById('scholarship').value = '1';
    document.getElementById('distance_km').value = '18.5';
    document.getElementById('clinic_department').value = 'General Practice';
    document.getElementById('appointment_slot').value = 'Afternoon';
    document.getElementById('prior_appointments').value = '3';
    document.getElementById('prior_noshows').value = '2';
    document.getElementById('sms_received').value = '0';
  } else if (presetType === 'low_risk') {
    // Same day / 1 day lead, senior citizen, perfect past record, SMS sent
    const apptDate = new Date();
    apptDate.setDate(now.getDate() + 1);
    apptInput.value = formatDate(apptDate);

    document.getElementById('patient_id').value = '102119';
    document.getElementById('age').value = '68';
    document.getElementById('gender').value = 'F';
    document.getElementById('scholarship').value = '0';
    document.getElementById('distance_km').value = '3.2';
    document.getElementById('clinic_department').value = 'Cardiology';
    document.getElementById('appointment_slot').value = 'Morning';
    document.getElementById('prior_appointments').value = '6';
    document.getElementById('prior_noshows').value = '0';
    document.getElementById('sms_received').value = '1';
  }
}

// Single Prediction Form Handler
async function handlePrediction(e) {
  e.preventDefault();
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.innerHTML = '<span>Evaluating Live Model...</span>';

  const payload = {
    patient_id: parseInt(document.getElementById('patient_id').value),
    age: parseFloat(document.getElementById('age').value),
    gender: document.getElementById('gender').value,
    scholarship: parseInt(document.getElementById('scholarship').value),
    distance_km: parseFloat(document.getElementById('distance_km').value),
    clinic_department: document.getElementById('clinic_department').value,
    appointment_slot: document.getElementById('appointment_slot').value,
    scheduled_day: document.getElementById('scheduled_day').value.replace('T', ' ') + ':00',
    appointment_day: document.getElementById('appointment_day').value,
    prior_appointments: parseInt(document.getElementById('prior_appointments').value),
    prior_noshows: parseInt(document.getElementById('prior_noshows').value),
    sms_received: parseInt(document.getElementById('sms_received').value)
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`HTTP Error: ${res.status}`);
    }

    const data = await res.json();
    renderPredictionResult(data);
  } catch (err) {
    alert('Error connecting to prediction service: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>Evaluate No-Show Probability</span>';
  }
}

function renderPredictionResult(data) {
  const badge = document.getElementById('riskBadge');
  badge.className = `badge badge-${data.badge_color}`;
  badge.textContent = data.risk_tier;

  const content = document.getElementById('resultsContent');

  let driversHtml = '';
  if (data.risk_drivers && data.risk_drivers.length > 0) {
    driversHtml = data.risk_drivers.map(d => {
      let pillClass = 'pill-neutral';
      if (d.effect.includes('Increases')) pillClass = 'pill-increase';
      if (d.effect.includes('Decreases')) pillClass = 'pill-decrease';

      return `
        <li class="driver-item">
          <div>
            <span class="driver-name">${d.factor}</span>
            <div class="driver-detail">${d.detail}</div>
          </div>
          <span class="driver-pill ${pillClass}">${d.effect}</span>
        </li>
      `;
    }).join('');
  }

  let scoreColor = '#10b981';
  if (data.no_show_probability >= 0.45) scoreColor = '#ef4444';
  else if (data.no_show_probability >= data.operational_threshold) scoreColor = '#ea580c';
  else if (data.no_show_probability >= 0.20) scoreColor = '#f59e0b';

  content.innerHTML = `
    <div class="score-display-box">
      <div class="score-main" style="color: ${scoreColor};">${data.risk_percentage}%</div>
      <div class="score-label">Predicted Probability of No-Show</div>
      <div class="threshold-notice">
        Decision Threshold: &tau; = ${data.operational_threshold} | Status: <strong>${data.prediction_label}</strong>
      </div>
    </div>

    <div class="action-banner">
      <div class="action-banner-title">Recommended Clinical Operations Action: ${data.recommended_action}</div>
      <div class="action-banner-desc">${data.operational_workflow}</div>
    </div>

    <div style="margin-bottom: 0.5rem;">
      <strong style="font-size: 0.85rem; color: #1e293b;">Key Influential Factors for this Patient:</strong>
    </div>
    <ul class="drivers-list">
      ${driversHtml}
    </ul>

    <div class="disclaimer-box">
      <strong>Human-in-the-Loop Safeguard:</strong> ${data.governance_notice}
    </div>
  `;
}

// Batch demonstration
async function loadSampleBatch() {
  const sampleAppointments = [
    { appointment_id: 2001, patient_id: 104501, age: 72, gender: 'F', scholarship: 0, distance_km: 2.5, clinic_department: 'Cardiology', scheduled_day: '2026-05-10 09:00:00', appointment_day: '2026-05-11', appointment_slot: 'Morning', prior_appointments: 5, prior_noshows: 0, sms_received: 1 },
    { appointment_id: 2002, patient_id: 104502, age: 24, gender: 'M', scholarship: 1, distance_km: 19.2, clinic_department: 'General Practice', scheduled_day: '2026-04-15 11:30:00', appointment_day: '2026-05-15', appointment_slot: 'Afternoon', prior_appointments: 2, prior_noshows: 2, sms_received: 0 },
    { appointment_id: 2003, patient_id: 104503, age: 41, gender: 'F', scholarship: 0, distance_km: 6.8, clinic_department: 'Orthopedics', scheduled_day: '2026-05-02 14:00:00', appointment_day: '2026-05-12', appointment_slot: 'Morning', prior_appointments: 1, prior_noshows: 0, sms_received: 1 },
    { appointment_id: 2004, patient_id: 104504, age: 8, gender: 'M', scholarship: 0, distance_km: 4.1, clinic_department: 'Pediatrics', scheduled_day: '2026-05-08 10:15:00', appointment_day: '2026-05-12', appointment_slot: 'Morning', prior_appointments: 3, prior_noshows: 0, sms_received: 1 },
    { appointment_id: 2005, patient_id: 104505, age: 29, gender: 'F', scholarship: 1, distance_km: 16.0, clinic_department: 'Dermatology', scheduled_day: '2026-04-20 16:45:00', appointment_day: '2026-05-18', appointment_slot: 'Afternoon', prior_appointments: 4, prior_noshows: 2, sms_received: 0 }
  ];

  try {
    const res = await fetch('/batch-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sampleAppointments)
    });

    const data = await res.json();
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    data.results.forEach(r => {
      let badgeClass = 'badge-low';
      if (r.risk_tier.includes('Critical')) badgeClass = 'badge-critical';
      else if (r.risk_tier.includes('High')) badgeClass = 'badge-high';
      else if (r.risk_tier.includes('Moderate')) badgeClass = 'badge-moderate';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>#${r.appointment_id}</strong></td>
        <td>${r.patient_id}</td>
        <td>${r.lead_days} days</td>
        <td><strong>${r.risk_percentage}%</strong></td>
        <td><span class="badge ${badgeClass}">${r.risk_tier}</span></td>
        <td>${r.recommended_action}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    alert('Batch evaluation error: ' + err.message);
  }
}

