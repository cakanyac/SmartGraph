import { Routes } from '@angular/router';
import { DashboardComponent } from './pages/dashboard/dashboard';
import { ParcelleComponent } from './pages/parcelle/parcelle';
import { RobotComponent } from './pages/robot/robot';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'parcelle', component: ParcelleComponent },
  { path: 'robot', component: RobotComponent }
];
