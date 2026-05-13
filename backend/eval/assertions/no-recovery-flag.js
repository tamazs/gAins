// Checks that recovery_flag is false (normal session, no overtraining)
module.exports = (output) => {
  try {
    const clean = output.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      .replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
    return JSON.parse(clean).recovery_flag === false;
  } catch (e) {
    return false;
  }
};
