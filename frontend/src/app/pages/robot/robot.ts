import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DataService, Robot } from '../../services/data.service';

@Component({
  selector: 'app-robot',
  imports: [CommonModule, FormsModule],
  templateUrl: './robot.html',
  styleUrl: './robot.css',
})
export class RobotComponent implements OnInit {
  robots: Robot[] = [];
  loading = true;
  showForm = false;
  isEditMode = false;
  
  formData = {
    modele: '',
    type: '',
    weed_detection_threshold: 50,
    statut: 'inactif' as 'actif' | 'inactif' | 'en_mission'
  };
  editingRobotId: string | null = null;

  constructor(private dataService: DataService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.dataService.getRobots().subscribe({
      next: (robots) => {
        this.robots = robots;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading robots:', err);
        this.loading = false;
      }
    });
  }

  getStatusColor(statut: string): string {
    switch (statut) {
      case 'actif':
        return '#28a745';
      case 'en_mission':
        return '#ffc107';
      case 'inactif':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  }

  getStatusLabel(statut: string): string {
    switch (statut) {
      case 'actif':
        return 'Actif';
      case 'en_mission':
        return 'En mission';
      case 'inactif':
        return 'Inactif';
      default:
        return 'Unknown';
    }
  }

  changeStatus(robot: Robot, newStatus: 'actif' | 'inactif' | 'en_mission'): void {
    this.dataService.updateRobot(robot.id, { ...robot, statut: newStatus }).subscribe({
      next: (updated) => {
        const index = this.robots.findIndex(r => r.id === robot.id);
        if (index !== -1) {
          this.robots[index] = updated;
          this.cdr.detectChanges();
        }
      },
      error: (err) => console.error('Error updating robot:', err)
    });
  }

  deleteRobot(robotId: string): void {
    if (confirm('Êtes-vous sûr?')) {
      this.dataService.deleteRobot(robotId).subscribe({
        next: () => {
          this.robots = this.robots.filter(r => r.id !== robotId);
          this.cdr.detectChanges();
        },
        error: (err) => console.error('Error deleting robot:', err)
      });
    }
  }

  openCreateForm(): void {
    this.isEditMode = false;
    this.editingRobotId = null;
    this.formData = { modele: '', type: '', weed_detection_threshold: 50, statut: 'inactif' };
    this.showForm = true;
  }

  openEditForm(robot: Robot): void {
    this.isEditMode = true;
    this.editingRobotId = robot.id;
    this.formData = { 
      modele: robot.modele, 
      type: robot.type, 
      weed_detection_threshold: robot.weed_detection_threshold,
      statut: robot.statut 
    };
    this.showForm = true;
  }

  saveRobot(): void {
    if (!this.formData.modele || !this.formData.type) {
      alert('Veuillez remplir tous les champs');
      return;
    }

    if (this.isEditMode && this.editingRobotId) {
      // Update existing robot
      this.dataService.updateRobot(this.editingRobotId, {
        modele: this.formData.modele,
        type: this.formData.type,
        weed_detection_threshold: this.formData.weed_detection_threshold,
        statut: this.formData.statut
      }).subscribe({
        next: (updated) => {
          const index = this.robots.findIndex(r => r.id === this.editingRobotId);
          if (index !== -1) {
            this.robots[index] = updated;
          }
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Robot modifié avec succès !');
        },
        error: (err) => {
          console.error('Error updating robot:', err);
          alert('Erreur lors de la modification');
        }
      });
    } else {
      // Create new robot
      this.dataService.createRobot({
        modele: this.formData.modele,
        type: this.formData.type,
        weed_detection_threshold: this.formData.weed_detection_threshold,
        statut: this.formData.statut
      }).subscribe({
        next: (robot) => {
          this.robots.push(robot);
          this.showForm = false;
          this.cdr.detectChanges();
          alert('Robot créé avec succès !');
        },
        error: (err) => {
          console.error('Error creating robot:', err);
          alert('Erreur lors de la création');
        }
      });
    }
  }

  cancelForm(): void {
    this.showForm = false;
  }
}
