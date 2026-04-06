/* ═══════════════════════════════════════════════════════════════
   NutriWise — app.js
   Handles: multi-step form, API calls, Chart.js rendering,
            meal tabs, food search, weekly projection table
   ═══════════════════════════════════════════════════════════════ */

"use strict";

// ─── State ────────────────────────────────────────────────────────────────────
const State = {
  currentStep: 1,
  totalSteps: 4,
  selectedGoal: null,
  plan: null,
  charts: {},
};

const API_BASE = window.location.origin;

function formatApiError(detail) {
  if (!detail) return 'Server error';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => (d && typeof d === 'object' && d.msg ? d.msg : String(d)))
      .filter(Boolean);
    return msgs.length ? msgs.join('; ') : 'Validation error';
  }
  if (typeof detail === 'object') {
    if (detail.msg) return detail.msg;
    try { return JSON.stringify(detail); } catch (_) { return 'Server error'; }
  }
  return String(detail);
}

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  bindGoalCards();
  bindNavPills();
  bindHealthCondition();
  updateStepUI();
  $('#searchBtn')?.addEventListener('click', runFoodSearch);
  $('#searchInput')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runFoodSearch();
  });
});

function bindHealthCondition() {
  const condition = $('#health_condition');
  const otherGroup = $('#healthOtherGroup');
  if (!condition || !otherGroup) return;
  const toggle = () => {
    otherGroup.classList.toggle('hidden', condition.value !== 'other');
  };
  condition.addEventListener('change', toggle);
  toggle();
}

// ─── Step navigation ──────────────────────────────────────────────────────────
function goStep(n) {
  if (n < 1 || n > State.totalSteps) return;
  // Validate before going forward
  if (n > State.currentStep && !validateStep(State.currentStep)) return;

  State.currentStep = n;
  updateStepUI();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateStepUI() {
  $$('.step-pane').forEach((pane) => pane.classList.add('hidden'));
  const active = $(`#step${State.currentStep}`);
  if (active) {
    active.classList.remove('hidden');
    active.style.animation = 'none';
    requestAnimationFrame(() => { active.style.animation = ''; });
  }

  $$('.step-item').forEach((item, i) => {
    const stepNum = i + 1;
    item.classList.remove('active', 'done');
    if (stepNum === State.currentStep) item.classList.add('active');
    else if (stepNum < State.currentStep) item.classList.add('done');
    item.querySelector('.step-circle').textContent =
      stepNum < State.currentStep ? '✓' : stepNum;
  });
}

// ─── Validation ───────────────────────────────────────────────────────────────
function validateStep(step) {
  const errors = [];
  if (step === 1) {
    const age = parseInt($('#age')?.value);
    const height = parseFloat($('#height_cm')?.value);
    const weight = parseFloat($('#weight_kg')?.value);
    const sex = $('#sex')?.value;
    if (!age || age < 10 || age > 100) errors.push('Please enter a valid age (10–100)');
    if (!sex) errors.push('Please select your sex');
    if (!height || height < 100 || height > 250) errors.push('Please enter a valid height (100–250 cm)');
    if (!weight || weight < 20 || weight > 300) errors.push('Please enter a valid weight (20–300 kg)');
  }
  if (step === 2) {
    if (!State.selectedGoal) errors.push('Please select your fitness goal');
  }
  if (step === 3) {
    const activity = $('#activity_level')?.value;
    if (!activity) errors.push('Please select your activity level');
  }
  if (errors.length > 0) {
    showToast(errors[0], 'error');
    return false;
  }
  return true;
}

// ─── Goal cards ───────────────────────────────────────────────────────────────
function bindGoalCards() {
  $$('.goal-card').forEach((card) => {
    card.addEventListener('click', () => {
      $$('.goal-card').forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');
      State.selectedGoal = card.dataset.goal;
    });
  });
}

// ─── Nav pills (scroll to section) ───────────────────────────────────────────
function bindNavPills() {
  $$('.nav-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      $$('.nav-pill').forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
    });
  });
}

// ─── Collect form data ────────────────────────────────────────────────────────
function collectFormData() {
  const v = (id) => document.getElementById(id)?.value?.trim() || '';
  const n = (id) => parseFloat(document.getElementById(id)?.value) || null;
  const i = (id) => parseInt(document.getElementById(id)?.value) || null;

  const healthCondition = v('health_condition');
  const healthOther = v('health_notes_other');
  const healthNotes = healthCondition === 'other'
    ? (healthOther || null)
    : (healthCondition || null);

  return {
    age:             i('age'),
    sex:             v('sex'),
    height_cm:       n('height_cm'),
    weight_kg:       n('weight_kg'),
    body_fat_pct:    n('body_fat_pct') || null,
    activity_level:  v('activity_level'),
    sleep_h:         n('sleep_h') || 7,
    meals_per_day:   i('meals_per_day') || 4,
    goal:            State.selectedGoal,
    target_weight_kg: n('target_weight_kg') || null,
    timeline_weeks:  i('timeline_weeks') || 12,
    diet_type:       v('diet_type') || 'none',
    cuisine:         v('cuisine') || 'india',
    allergies:       v('allergies') || null,
    health_notes:    healthNotes,
  };
}

// ─── Generate plan ────────────────────────────────────────────────────────────
async function generatePlan() {
  if (!validateStep(3)) return;

  const data = collectFormData();

  // Show loading
  State.currentStep = 4;
  updateStepUI();
  showLoadingScreen();

  const steps = [
    'Calculating BMR & TDEE from your profile...',
    'Running macro recommendation model...',
    'Scoring foods for your goal...',
    'Building personalized meal plan...',
    'Generating weekly projections...',
  ];
  animateLoadingSteps(steps);

  try {
    const resp = await fetch(`${API_BASE}/api/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!resp.ok) {
      let payload = null;
      try {
        payload = await resp.json();
      } catch (_) {
        throw new Error(`Server error (${resp.status})`);
      }
      throw new Error(formatApiError(payload?.detail) || `Server error (${resp.status})`);
    }

    State.plan = await resp.json();
    renderResults(State.plan, data);
  } catch (err) {
    hideLoadingScreen();
    showToast(`Error: ${err.message}`, 'error');
    State.currentStep = 3;
    updateStepUI();
  }
}

// ─── Loading screen helpers ───────────────────────────────────────────────────
function showLoadingScreen() {
  $('#loadingScreen')?.classList.remove('hidden');
  $('#resultsScreen')?.classList.add('hidden');
}
function hideLoadingScreen() {
  $('#loadingScreen')?.classList.add('hidden');
}

function animateLoadingSteps(steps) {
  const container = $('#loadingSteps');
  if (!container) return;
  container.innerHTML = steps.map((s, i) =>
    `<div class="loading-step" id="ls${i}"><div class="step-dot"></div>${s}</div>`
  ).join('');

  let idx = 0;
  const interval = setInterval(() => {
    if (idx > 0) $(`#ls${idx - 1}`)?.classList.replace('active', 'done');
    $(`#ls${idx}`)?.classList.add('active');
    idx++;
    if (idx >= steps.length) clearInterval(interval);
  }, 700);
}

// ─── Render results ───────────────────────────────────────────────────────────
function renderResults(plan, profile) {
  hideLoadingScreen();
  $('#resultsScreen')?.classList.remove('hidden');

  renderHeader(plan);
  renderMetrics(plan);
  renderMacroBar(plan.macros);
  renderMacroChart(plan.macros);
  renderWeeklyChart(plan.weekly_projections);
  renderMeals(plan.meals);
  renderWeeklyTable(plan.weekly_projections);
  renderTips(plan.tips, plan.warnings);
  renderSupplements(plan.supplements);
}

function renderHeader(plan) {
  const goalLabels = { loss: 'Fat Loss', muscle: 'Muscle Gain', gain: 'Healthy Bulk' };
  const badge = $('#goalBadge');
  const title = $('#resultsTitle');
  const sub   = $('#resultsSub');
  if (badge) badge.textContent = goalLabels[plan.goal_code] || plan.goal_label;
  if (title) title.textContent = `Your ${plan.goal_label} Plan`;
  if (sub) sub.textContent =
    `Personalized for your ${plan.bmi_category} BMI of ${plan.bmi} | Powered by NutriWise AI`;
}

function renderMetrics(plan) {
  const metrics = [
    { label: 'Daily Calories', value: `${Math.round(plan.calorie_target)} kcal`, cls: 'amber' },
    { label: 'Protein',        value: `${Math.round(plan.macros.protein_g)}g`,   cls: '' },
    { label: 'Carbohydrates',  value: `${Math.round(plan.macros.carbs_g)}g`,     cls: '' },
    { label: 'Fat',            value: `${Math.round(plan.macros.fat_g)}g`,       cls: '' },
    { label: 'BMR',            value: `${Math.round(plan.bmr)} kcal`,            cls: 'cream' },
    { label: 'TDEE',           value: `${Math.round(plan.tdee)} kcal`,           cls: 'cream' },
    { label: 'BMI',            value: `${plan.bmi}`,                             cls: '' },
    { label: 'Water / Day',    value: `${(plan.hydration_ml / 1000).toFixed(1)}L`, cls: '' },
  ];

  const grid = $('#metricsGrid');
  if (!grid) return;
  grid.innerHTML = metrics.map((m) =>
    `<div class="metric-card">
      <div class="metric-value ${m.cls}">${m.value}</div>
      <div class="metric-label">${m.label}</div>
    </div>`
  ).join('');
}

function renderMacroBar(macros) {
  const bar = $('#macroBar');
  const legend = $('#macroLegend');
  if (!bar) return;

  bar.innerHTML = `
    <div class="macro-seg" style="width:${macros.protein_pct}%;background:#52A852"></div>
    <div class="macro-seg" style="width:${macros.carbs_pct}%;background:#F59E0B"></div>
    <div class="macro-seg" style="width:${macros.fat_pct}%;background:#4A90D9"></div>`;

  if (legend) legend.innerHTML = `
    <div class="legend-item"><div class="legend-dot" style="background:#52A852"></div>Protein ${macros.protein_pct}% (${Math.round(macros.protein_g)}g)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#F59E0B"></div>Carbs ${macros.carbs_pct}% (${Math.round(macros.carbs_g)}g)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4A90D9"></div>Fat ${macros.fat_pct}% (${Math.round(macros.fat_g)}g)</div>`;
}

function renderMacroChart(macros) {
  const ctx = document.getElementById('macroChart');
  if (!ctx) return;
  if (State.charts.macro) State.charts.macro.destroy();

  State.charts.macro = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Protein', 'Carbohydrates', 'Fat'],
      datasets: [{
        data: [
          Math.round(macros.protein_g * 4),
          Math.round(macros.carbs_g * 4),
          Math.round(macros.fat_g * 9),
        ],
        backgroundColor: ['#52A852', '#F59E0B', '#4A90D9'],
        borderWidth: 0,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.label}: ${c.raw} kcal`,
          },
          backgroundColor: '#1A2E1A',
          titleColor: '#C8D8C8',
          bodyColor: '#C8D8C8',
          borderColor: 'rgba(138,188,138,.2)',
          borderWidth: 1,
        },
      },
    },
  });
}

function renderWeeklyChart(projections) {
  const ctx = document.getElementById('weeklyChart');
  if (!ctx || !projections?.length) return;
  if (State.charts.weekly) State.charts.weekly.destroy();

  const labels  = projections.map((p) => `Wk ${p.week}`);
  const weights = projections.map((p) => p.projected_weight_kg);
  const cals    = projections.map((p) => p.calorie_target);

  State.charts.weekly = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Calories',
          data: cals,
          backgroundColor: 'rgba(245,158,11,.25)',
          borderColor: '#F59E0B',
          borderWidth: 1.5,
          borderRadius: 4,
          yAxisID: 'y',
        },
        {
          label: 'Weight (kg)',
          data: weights,
          type: 'line',
          borderColor: '#52A852',
          backgroundColor: 'rgba(82,168,82,.08)',
          pointRadius: 4,
          pointBackgroundColor: '#52A852',
          yAxisID: 'y1',
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1A2E1A',
          titleColor: '#C8D8C8',
          bodyColor: '#C8D8C8',
          borderColor: 'rgba(138,188,138,.2)',
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          ticks: { color: '#7A9A7A', font: { size: 10 } },
          grid: { color: 'rgba(138,188,138,.06)' },
        },
        y: {
          position: 'left',
          ticks: { color: '#7A9A7A', font: { size: 10 } },
          grid: { color: 'rgba(138,188,138,.06)' },
        },
        y1: {
          position: 'right',
          ticks: { color: '#52A852', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

// ─── Meals ────────────────────────────────────────────────────────────────────
function renderMeals(meals) {
  const tabBar  = $('#mealTabBar');
  const paneWrap = $('#mealPanes');
  if (!tabBar || !paneWrap || !meals?.length) return;

  tabBar.innerHTML = meals.map((m, i) =>
    `<button class="meal-tab ${i === 0 ? 'active' : ''}" onclick="switchMeal(${i})">${m.meal_name}</button>`
  ).join('');

  paneWrap.innerHTML = meals.map((meal, i) => {
    const totalMacros = meal.foods.reduce(
      (acc, f) => ({ p: acc.p + (f.protein_g || 0), c: acc.c + (f.carbs_g || 0), f: acc.f + (f.fat_g || 0) }),
      { p: 0, c: 0, f: 0 }
    );
    return `
      <div class="meal-pane ${i === 0 ? 'active' : ''}" id="mealPane${i}">
        <div class="meal-header-row">
          <div>
            <div class="meal-name-badge">${meal.meal_name}</div>
            <div class="meal-time">${meal.time_window}</div>
          </div>
          <div style="text-align:right">
            <div class="meal-total-cals">${Math.round(meal.total_calories)} kcal</div>
            <div style="font-size:.7rem;color:var(--text3)">P:${Math.round(totalMacros.p)}g | C:${Math.round(totalMacros.c)}g | F:${Math.round(totalMacros.f)}g</div>
          </div>
        </div>
        <table class="food-table">
          <thead>
            <tr>
              <th>Food</th>
              <th style="text-align:right">Qty</th>
              <th style="text-align:right">kcal</th>
              <th>Macros</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            ${meal.foods.map((f) => `
              <tr>
                <td class="food-name-cell">
                  ${f.name}
                  ${f.prep_note ? `<span class="food-note">${f.prep_note}</span>` : ''}
                </td>
                <td style="text-align:right;color:var(--text3)">${Math.round(f.quantity_g || 0)}g</td>
                <td style="text-align:right">${Math.round(f.calories)}</td>
                <td>
                  <span class="macro-tag tag-p">P${Math.round(f.protein_g)}g</span>
                  <span class="macro-tag tag-c">C${Math.round(f.carbs_g)}g</span>
                  <span class="macro-tag tag-f">F${Math.round(f.fat_g)}g</span>
                </td>
                <td>
                  <div class="score-bar">
                    <div class="score-track">
                      <div class="score-fill" style="width:${f.score || 0}%;background:${scoreColor(f.score || 0)}"></div>
                    </div>
                    <span class="score-num">${Math.round(f.score || 0)}</span>
                  </div>
                </td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }).join('');
}

function switchMeal(idx) {
  $$('.meal-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  $$('.meal-pane').forEach((p, i) => p.classList.toggle('active', i === idx));
}

function scoreColor(score) {
  if (score >= 70) return '#52A852';
  if (score >= 45) return '#F59E0B';
  return '#EF4444';
}

// ─── Weekly projection table ──────────────────────────────────────────────────
function renderWeeklyTable(projections) {
  const tbody = $('#weeklyTableBody');
  if (!tbody || !projections?.length) return;

  const startWeight = projections[0]?.projected_weight_kg;
  tbody.innerHTML = projections.map((p) => {
    const delta = (p.projected_weight_kg - startWeight).toFixed(1);
    const sign = delta > 0 ? '+' : '';
    return `<tr>
      <td class="week-num">Week ${p.week}</td>
      <td>${Math.round(p.calorie_target)} kcal</td>
      <td>${Math.round(p.protein_g)}g</td>
      <td>${Math.round(p.carbs_g)}g</td>
      <td>${Math.round(p.fat_g)}g</td>
      <td>${p.projected_weight_kg} kg <span class="weight-change">${sign}${delta}</span></td>
    </tr>`;
  }).join('');
}

// ─── Tips & warnings ──────────────────────────────────────────────────────────
function renderTips(tips, warnings) {
  const container = $('#tipsContainer');
  if (!container) return;

  const warningHTML = (warnings || []).map((w) =>
    `<div class="tip-item warning-item"><div class="tip-bullet"></div>${w}</div>`
  ).join('');

  const tipsHTML = (tips || []).map((t) =>
    `<div class="tip-item"><div class="tip-bullet"></div>${t}</div>`
  ).join('');

  container.innerHTML = warningHTML + tipsHTML;
}

function renderSupplements(supplements) {
  const container = $('#suppContainer');
  if (!container) return;

  container.innerHTML = (supplements || []).map((s) => {
    const parts = s.split(' - ').length > 1 ? s.split(' - ') : s.split(' — ');
    const [name, ...rest] = parts;
    return `<div class="supp-card">
      <div class="supp-icon">S</div>
      <div>
        <div style="color:var(--cream);font-weight:500;font-size:.82rem">${name}</div>
        ${rest.length ? `<div style="color:var(--text3);font-size:.72rem;margin-top:2px">${rest.join(' - ')}</div>` : ''}
      </div>
    </div>`;
  }).join('');
}

// ─── Food search ──────────────────────────────────────────────────────────────
async function runFoodSearch() {
  const q    = $('#searchInput')?.value?.trim();
  const goal = State.plan?.goal_code || State.selectedGoal || 'muscle';
  if (!q || q.length < 2) return;

  const resultsWrap = $('#searchResults');
  if (resultsWrap) resultsWrap.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:.5rem">Searching...</div>';

  try {
    const resp = await fetch(`${API_BASE}/api/foods/search?q=${encodeURIComponent(q)}&goal=${goal}&limit=15`);
    if (!resp.ok) throw new Error('Search failed');
    const foods = await resp.json();
    renderSearchResults(foods);
  } catch (err) {
    if (resultsWrap) resultsWrap.innerHTML = `<div style="color:#FCA5A5;font-size:.8rem;padding:.5rem">Error: ${err.message}</div>`;
  }
}

function renderSearchResults(foods) {
  const container = $('#searchResults');
  if (!container) return;

  if (!foods.length) {
    container.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:.5rem">No results found</div>';
    return;
  }

  container.innerHTML = foods.map((f) => `
    <div class="search-result-item">
      <div>
        <div class="search-food-name">${f.name}</div>
        <div class="search-food-cat">${f.category} | ${f.source}</div>
      </div>
      <div class="search-food-macros">
        <span class="macro-tag tag-p">P${Math.round(f.protein_g)}g</span>
        <span class="macro-tag tag-c">C${Math.round(f.carbs_g)}g</span>
        <span class="macro-tag tag-f">F${Math.round(f.fat_g)}g</span>
        <span style="font-size:.7rem;color:var(--text3);align-self:center">${Math.round(f.calories)} kcal</span>
        <div class="score-bar" style="width:70px">
          <div class="score-track">
            <div class="score-fill" style="width:${f.score || 0}%;background:${scoreColor(f.score || 0)}"></div>
          </div>
          <span class="score-num">${Math.round(f.score || 0)}</span>
        </div>
      </div>
    </div>`).join('');
}

// ─── Reset / print ────────────────────────────────────────────────────────────
function resetApp() {
  State.currentStep = 1;
  State.selectedGoal = null;
  State.plan = null;
  Object.values(State.charts).forEach((c) => c?.destroy?.());
  State.charts = {};

  $$('.goal-card').forEach((c) => c.classList.remove('selected'));
  $('#resultsScreen')?.classList.add('hidden');
  updateStepUI();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function printPlan() { window.print(); }

// ─── Toast notifications ──────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const existing = $('.nw-toast');
  if (existing) existing.remove();

  const colors = {
    info: '#52A852',
    error: '#EF4444',
    warning: '#F59E0B',
  };

  const toast = document.createElement('div');
  toast.className = 'nw-toast';
  toast.style.cssText = `
    position:fixed; bottom:1.5rem; right:1.5rem; z-index:9999;
    background:var(--forest3); border:1px solid ${colors[type]};
    color:var(--cream); padding:.75rem 1.25rem; border-radius:10px;
    font-size:.85rem; font-family:var(--font-body);
    box-shadow:0 8px 24px rgba(0,0,0,.4);
    animation: slideIn .3s ease both;
    max-width: 340px;
  `;
  toast.textContent = msg;

  const style = document.createElement('style');
  style.textContent = `@keyframes slideIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }`;
  document.head.appendChild(style);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ─── Expose to HTML ───────────────────────────────────────────────────────────
window.goStep = goStep;
window.generatePlan = generatePlan;
window.resetApp = resetApp;
window.printPlan = printPlan;
window.switchMeal = switchMeal;
window.runFoodSearch = runFoodSearch;
