// Checks that reasoning references at least 2 established training science terms
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    const d = JSON.parse(clean);
    const text = JSON.stringify(d).toLowerCase();
    const terms = [
      'progressive overload', 'overload', 'rir', 'reps in reserve',
      'volume', 'hypertrophy', 'progression', 'deload',
      'rep range', 'fatigue', 'recovery', 'load',
    ];
    return terms.filter(t => text.includes(t)).length >= 2;
  } catch (e) {
    return false;
  }
};
