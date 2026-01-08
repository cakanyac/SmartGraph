import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface Parcelle {
  id: string;
  nom: string;
  superficie: number;
  culture: string;
  type_sol: string;
  latitude: number;
  longitude: number;
}

export interface Capteur {
  id: string;
  parcelle_id: string;
  type: string;
  statut: string;
}

export interface Robot {
  id: string;
  modele: string;
  type: string;
  weed_detection_threshold: number;
  statut: 'actif' | 'inactif' | 'en_mission';
}

export interface Observation {
  id: string;
  capteur_id: string;
  valeur: number;
  timestamp: string;
  qualite: string;
}

export interface Alerte {
  id: string;
  parcelle_id: string;
  type: string;
  message: string;
  severite: 'basse' | 'moyenne' | 'haute' | 'critique';
  lue: boolean;
  timestamp: string;
}

export interface DashboardStats {
  parcelles: number;
  capteurs: number;
  robots: number;
  alertes: number;
  missionsEnCours: number;
  capteursDysfonctionne: number;
}

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = 'http://localhost:8080/api/v1';

  constructor(private http: HttpClient) {}

  // PARCELLES
  getParcelles(): Observable<Parcelle[]> {
    return this.http.get<Parcelle[]>(`${this.apiUrl}/parcelles`);
  }

  getParcelleById(id: string): Observable<Parcelle> {
    return this.http.get<Parcelle>(`${this.apiUrl}/parcelles/${id}`);
  }

  createParcelle(parcelle: Omit<Parcelle, 'id'>): Observable<Parcelle> {
    return this.http.post<Parcelle>(`${this.apiUrl}/parcelles`, parcelle);
  }

  updateParcelle(id: string, parcelle: Partial<Parcelle>): Observable<Parcelle> {
    return this.http.put<Parcelle>(`${this.apiUrl}/parcelles/${id}`, parcelle);
  }

  deleteParcelle(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/parcelles/${id}`);
  }

  // CAPTEURS
  getCapteurs(): Observable<Capteur[]> {
    return this.http.get<Capteur[]>(`${this.apiUrl}/capteurs`);
  }

  getCapteursByParcelleId(parcelleId: string): Observable<Capteur[]> {
    return this.http.get<Capteur[]>(`${this.apiUrl}/capteurs`).pipe(
      map(capteurs => capteurs.filter(c => c.parcelle_id === parcelleId))
    );
  }

  createCapteur(capteur: Omit<Capteur, 'id'>): Observable<Capteur> {
    return this.http.post<Capteur>(`${this.apiUrl}/capteurs`, capteur);
  }

  updateCapteur(id: number, capteur: Partial<Capteur>): Observable<Capteur> {
    return this.http.put<Capteur>(`${this.apiUrl}/capteurs/${id}`, capteur);
  }

  deleteCapteur(id: number): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/capteurs/${id}`);
  }

  // ROBOTS
  getRobots(): Observable<Robot[]> {
    return this.http.get<Robot[]>(`${this.apiUrl}/robots`);
  }

  getRobotById(id: string): Observable<Robot> {
    return this.http.get<Robot>(`${this.apiUrl}/robots/${id}`);
  }

  createRobot(robot: Omit<Robot, 'id'>): Observable<Robot> {
    return this.http.post<Robot>(`${this.apiUrl}/robots`, robot);
  }

  updateRobot(id: string, robot: Partial<Robot>): Observable<Robot> {
    return this.http.put<Robot>(`${this.apiUrl}/robots/${id}`, robot);
  }

  deleteRobot(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/robots/${id}`);
  }

  // DASHBOARD - À DÉFINIR AVEC MAËL
  // Pour l'instant, retourner les stats en comptant les entités
  getDashboardStats(): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(`${this.apiUrl}/dashboard/stats`);
  }

  // OBSERVATIONS
  getObservations(): Observable<Observation[]> {
    return this.http.get<Observation[]>(`${this.apiUrl}/observations`);
  }

  getObservationById(id: string): Observable<Observation> {
    return this.http.get<Observation>(`${this.apiUrl}/observations/${id}`);
  }

  getObservationsByCapteurId(capteurId: string): Observable<Observation[]> {
    return this.http.get<Observation[]>(`${this.apiUrl}/observations`).pipe(
      map(obs => obs.filter(o => o.capteur_id === capteurId))
    );
  }

  createObservation(observation: Omit<Observation, 'id'>): Observable<Observation> {
    return this.http.post<Observation>(`${this.apiUrl}/observations`, observation);
  }

  updateObservation(id: string, observation: Partial<Observation>): Observable<Observation> {
    return this.http.put<Observation>(`${this.apiUrl}/observations/${id}`, observation);
  }

  deleteObservation(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/observations/${id}`);
  }

  // ALERTES
  getAlertes(): Observable<Alerte[]> {
    return this.http.get<Alerte[]>(`${this.apiUrl}/alertes`);
  }

  getAlerteById(id: string): Observable<Alerte> {
    return this.http.get<Alerte>(`${this.apiUrl}/alertes/${id}`);
  }

  createAlerte(alerte: Omit<Alerte, 'id'>): Observable<Alerte> {
    return this.http.post<Alerte>(`${this.apiUrl}/alertes`, alerte);
  }

  markAlerteAsRead(id: string): Observable<Alerte> {
    return this.http.put<Alerte>(`${this.apiUrl}/alertes/${id}/marquer-lue`, {});
  }

  deleteAlerte(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/alertes/${id}`);
  }
}
