import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DataService, Observation } from '../../services/data.service';

@Component({
  selector: 'app-observations',
  imports: [CommonModule, FormsModule],
  templateUrl: './observations.html',
  styleUrl: './observations.css',
})
export class ObservationsComponent implements OnInit {
  observations: Observation[] = [];
  selectedObservation: Observation | null = null;
  loading = true;
  showForm = false;
  isEditMode = false;
  
  formData = {
    capteur_id: '',
    valeur: 0,
    timestamp: new Date().toISOString(),
    qualite: 'bonne'
  };

  constructor(private dataService: DataService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.loadObservations();
  }

  loadObservations(): void {
    this.loading = true;
    this.dataService.getObservations().subscribe({
      next: (observations) => {
        this.observations = observations;
        if (observations.length > 0) {
          this.selectedObservation = observations[0];
        }
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading observations:', err);
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  selectObservation(observation: Observation): void {
    this.selectedObservation = observation;
    this.showForm = false;
    this.cdr.detectChanges();
  }

  openCreateForm(): void {
    this.isEditMode = false;
    this.formData = { capteur_id: '', valeur: 0, timestamp: new Date().toISOString(), qualite: 'bonne' };
    this.showForm = true;
    this.cdr.detectChanges();
  }

  openEditForm(): void {
    if (!this.selectedObservation) return;
    this.isEditMode = true;
    this.formData = {
      capteur_id: this.selectedObservation.capteur_id,
      valeur: this.selectedObservation.valeur,
      timestamp: this.selectedObservation.timestamp,
      qualite: this.selectedObservation.qualite
    };
    this.showForm = true;
    this.cdr.detectChanges();
  }

  saveObservation(): void {
    if (!this.formData.capteur_id || this.formData.valeur === null) {
      alert('Veuillez remplir tous les champs');
      return;
    }

    if (this.isEditMode && this.selectedObservation) {
      this.dataService.updateObservation(this.selectedObservation.id, this.formData).subscribe({
        next: (updated) => {
          const index = this.observations.findIndex(o => o.id === this.selectedObservation!.id);
          if (index !== -1) {
            this.observations[index] = updated;
            this.selectedObservation = updated;
          }
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Observation modifiée avec succès !');
        },
        error: (err) => {
          console.error('Error updating observation:', err);
          alert('Erreur lors de la modification');
        }
      });
    } else {
      this.dataService.createObservation(this.formData).subscribe({
        next: (observation) => {
          this.observations.push(observation);
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Observation créée avec succès !');
        },
        error: (err) => {
          console.error('Error creating observation:', err);
          alert('Erreur lors de la création');
        }
      });
    }
  }

  deleteObservation(): void {
    if (!this.selectedObservation) return;
    if (confirm('Êtes-vous sûr de vouloir supprimer cette observation ?')) {
      this.dataService.deleteObservation(this.selectedObservation.id).subscribe({
        next: () => {
          this.observations = this.observations.filter(o => o.id !== this.selectedObservation!.id);
          this.selectedObservation = null;
          this.cdr.detectChanges();
          alert('Observation supprimée avec succès !');
        },
        error: (err) => {
          console.error('Error deleting observation:', err);
          alert('Erreur lors de la suppression');
        }
      });
    }
  }

  cancelForm(): void {
    this.showForm = false;
    this.cdr.detectChanges();
  }
}