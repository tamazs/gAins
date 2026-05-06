import { chromium } from '@playwright/test';

const browser = await chromium.launch();
const AUTH = JSON.stringify({ token: 'mock-token', userId: 'user-123', username: 'ironlifter' });

async function shot(url, name) {
  // Each page gets a fresh context with pre-seeded localStorage
  const ctx = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    storageState: {
      cookies: [],
      origins: [
        {
          origin: 'http://localhost:5173',
          localStorage: [{ name: 'auth', value: AUTH }],
        },
      ],
    },
  });
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `screenshots/${name}.png`, fullPage: false });
  console.log(`✓ ${name}`);
  await ctx.close();
}

await shot('http://localhost:5173/login', 'login');
await shot('http://localhost:5173/register', 'register');
await shot('http://localhost:5173/dashboard', 'dashboard');
await shot('http://localhost:5173/sessions', 'sessions');
await shot('http://localhost:5173/sessions/new', 'new-session');
await shot('http://localhost:5173/goals', 'goals');

await browser.close();
