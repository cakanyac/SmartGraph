import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DataService, Parcelle, Capteur } from '../../services/data.service';

@Component({
  selector: 'app-parcelle',
  imports: [CommonModule, FormsModule],
  templateUrl: './parcelle.html',
  styleUrl: './parcelle.css',
})
export class ParcelleComponent implements OnInit {
  parcelles: Parcelle[] = [];
  capteurs: Capteur[] = [];
  selectedParcelle: Parcelle | null = null;
  loading = true;
  showForm = false;
  isEditMode = false;
  
  formData = {
    nom: '',
    superficie: 0,
    culture: '',
    type_sol: '',
    latitude: 0,
    longitude: 0
  };

  constructor(private dataService: DataService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    console.log('ParcelleComponent initialized');
    this.dataService.getParcelles().subscribe({
      next: (parcelles) => {
        console.log('Parcelles loaded:', parcelles);
        this.parcelles = parcelles;
        if (parcelles.length > 0) {
          this.selectedParcelle = parcelles[0];
          this.loadCapteurs();
        }
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading parcelles:', err);
        alert('Erreur API: ' + err.message);
        this.loading = false;
      }
    });
  }

  selectParcelle(parcelle: Parcelle): void {
    this.selectedParcelle = parcelle;
    this.loadCapteurs();
    this.showForm = false;
    this.cdr.detectChanges();
  }

  private loadCapteurs(): void {
    if (!this.selectedParcelle) return;
    
    this.dataService.getCapteursByParcelleId(this.selectedParcelle.id).subscribe({
      next: (capteurs) => {
        this.capteurs = capteurs;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading capteurs:', err);
      }
    });
  }

  openCreateForm(): void {
    this.isEditMode = false;
    this.formData = { nom: '', superficie: 0, culture: '', type_sol: '', latitude: 0, longitude: 0 };
    this.showForm = true;
  }

  openEditForm(): void {
    if (!this.selectedParcelle) return;
    this.isEditMode = true;
    this.formData = {
      nom: this.selectedParcelle.nom,
      superficie: this.selectedParcelle.superficie,
      culture: this.selectedParcelle.culture,
      type_sol: this.selectedParcelle.type_sol,
      latitude: this.selectedParcelle.latitude,
      longitude: this.selectedParcelle.longitude
    };
    this.showForm = true;
  }

  createParcelle(): void {
    if (!this.formData.nom || this.formData.superficie <= 0) {
      alert('Veuillez remplir tous les champs correctement');
      return;
    }

    if (this.isEditMode && this.selectedParcelle) {
      // Update existing parcelle
      this.dataService.updateParcelle(this.selectedParcelle.id, {
        nom: this.formData.nom,
        superficie: this.formData.superficie,
        culture: this.formData.culture,
        type_sol: this.formData.type_sol,
        latitude: this.formData.latitude,
        longitude: this.formData.longitude
      }).subscribe({
        next: (updated) => {
          const index = this.parcelles.findIndex(p => p.id === this.selectedParcelle!.id);
          if (index !== -1) {
            this.parcelles[index] = updated;
            this.selectedParcelle = updated;
          }
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Parcelle modifiée avec succès !');
        },
        error: (err) => {
          console.error('Error updating parcelle:', err);
          alert('Erreur lors de la modification');
        }
      });
    } else {
      // Create new parcelle
      this.dataService.createParcelle({
        nom: this.formData.nom,
        superficie: this.formData.superficie,
        culture: this.formData.culture,
        type_sol: this.formData.type_sol,
        latitude: this.formData.latitude,
        longitude: this.formData.longitude
      }).subscribe({
        next: (parcelle) => {
          this.parcelles.push(parcelle);
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Parcelle créée avec succès !');
        },
        error: (err) => {
          console.error('Error creating parcelle:', err);
          alert('Erreur lors de la création');
        }
      });
    }
  }

  deleteParcelle(): void {
    if (!this.selectedParcelle) return;
    if (confirm('Êtes-vous sûr de vouloir supprimer cette parcelle ?')) {
      this.dataService.deleteParcelle(this.selectedParcelle.id).subscribe({
        next: () => {
          this.parcelles = this.parcelles.filter(p => p.id !== this.selectedParcelle!.id);
          this.selectedParcelle = null;
          this.capteurs = [];
          this.cdr.detectChanges();
          alert('Parcelle supprimée avec succès !');
        },
        error: (err) => {
          console.error('Error deleting parcelle:', err);
          alert('Erreur lors de la suppression');
        }
      });
    }
  }

  cancelForm(): void {
    this.showForm = false;
  }
}
