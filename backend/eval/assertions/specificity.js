// Checks that the response contains specific kg values in recommendations
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    const d = JSON.parse(clean);
    const text = JSON.stringify(d.exercise_advice).toLowerCase();
    return /\d+\.?\d*\s*kg/.test(text) ||
           d.exercise_advice.some(ea => ea.suggested_weight_kg != null);
  } catch (e) {
    return false;
  }
};
