import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService, Parcelle } from '../../services/data.service';

@Component({
  selector: 'app-map',
  imports: [CommonModule],
  templateUrl: './map.html',
  styleUrl: './map.css',
})
export class MapComponent implements OnInit {
  parcelles: Parcelle[] = [];
  selectedParcelle: Parcelle | null = null;
  loading = true;

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService.getParcelles().subscribe({
      next: (parcelles) => {
        this.parcelles = parcelles;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading parcelles:', err);
        this.loading = false;
      }
    });
  }

  selectParcelle(parcelle: Parcelle): void {
    this.selectedParcelle = parcelle;
  }
}
