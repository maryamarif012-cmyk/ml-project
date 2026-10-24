
import {
  Component,
  OnInit,
  ChangeDetectorRef
} from '@angular/core';

import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

interface PredictionResult {
  early_readmission_probability: number;
  interpretation: string;
  predicted_class: number;
  notice: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  sampleData: Record<string, any> | null = null;
  result: PredictionResult | null = null;
  errorMessage = '';

  loading = false;
  loadingSample = false;

  age = '[0-10)';
  gender = 'Female';
  admissionType = 1;
  inpatient = 0;
  emergency = 0;
  outpatient = 0;

  private readonly apiUrl = 'http://127.0.0.1:5000/api';

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.loadSampleData();
  }

  loadSampleData(): void {
    if (this.loadingSample) {
      return;
    }

    this.loadingSample = true;
    this.errorMessage = '';

    this.http
      .get<Record<string, any>>(`${this.apiUrl}/sample`)
      .subscribe({
        next: (data) => {
          this.sampleData = data;
          this.loadingSample = false;

          console.log('Sample data loaded:', data);
          this.cdr.detectChanges();
        },
        error: (err) => {
          this.sampleData = null;
          this.loadingSample = false;

          this.errorMessage =
            'Could not load sample data. Make sure Flask is running.';

          console.error('Sample request failed:', err);
          this.cdr.detectChanges();
        }
      });
  }

  predict(): void {
    if (this.loading || this.loadingSample) {
      return;
    }

    this.errorMessage = '';

    if (!this.sampleData) {
      this.loadSampleData();
      this.errorMessage =
        'Sample data is unavailable. Please wait for loading to finish.';
      return;
    }

    const patient = {
      ...this.sampleData,
      age: this.age,
      gender: this.gender,
      admission_type_id: Number(this.admissionType),
      number_inpatient: Number(this.inpatient),
      number_emergency: Number(this.emergency),
      number_outpatient: Number(this.outpatient)
    };

    this.loading = true;
    this.result = null;

    console.log('Sending prediction request:', patient);
    this.cdr.detectChanges();

    this.http
      .post<PredictionResult>(
        `${this.apiUrl}/predict`,
        patient
      )
      .subscribe({
        next: (data) => {
          this.result = data;
          this.loading = false;

          console.log('Prediction result:', data);
          console.log('Loading state:', this.loading);

          this.cdr.detectChanges();
        },

        error: (err) => {
          this.loading = false;

          this.errorMessage =
            err.error?.error ||
            `Prediction failed. HTTP status: ${err.status || 'unknown'}.`;

          console.error('Prediction request failed:', err);
          this.cdr.detectChanges();
        },

        complete: () => {
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }
}
