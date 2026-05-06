import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { PrivateRoute } from '../components/layout/PrivateRoute';
import { LoginPage } from '../pages/LoginPage';
import { RegisterPage } from '../pages/RegisterPage';
import { DashboardPage } from '../pages/DashboardPage';
import { SessionsPage } from '../pages/SessionsPage';
import { SessionDetailPage } from '../pages/SessionDetailPage';
import { NewSessionPage } from '../pages/NewSessionPage';
import { GoalsPage } from '../pages/GoalsPage';

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Protected */}
        <Route element={<PrivateRoute />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/sessions/new" element={<NewSessionPage />} />
          <Route path="/sessions/:id" element={<SessionDetailPage />} />
          <Route path="/goals" element={<GoalsPage />} />
        </Route>

        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
