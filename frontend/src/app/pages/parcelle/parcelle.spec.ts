import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Parcelle } from './parcelle';

describe('Parcelle', () => {
  let component: Parcelle;
  let fixture: ComponentFixture<Parcelle>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Parcelle]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Parcelle);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
