import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService, DashboardStats, Parcelle } from '../../services/data.service';


@Component({
  selector: 'app-dashboard',
  imports: [CommonModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css',
})
export class DashboardComponent implements OnInit {
  stats: DashboardStats | null = null;
  parcelles: Parcelle[] = [];
  loading = true;

  constructor(private dataService: DataService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.dataService.getDashboardStats().subscribe({
      next: (stats) => {
        this.stats = stats;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading dashboard stats:', err);
      }
    });

    this.dataService.getParcelles().subscribe({
      next: (parcelles) => {
        this.parcelles = parcelles;
        this.loading = false;
         this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error loading parcelles:', err);
        this.loading = false;
      }
    });
  }
}
