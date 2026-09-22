# BINN / UDE — diagnostic-only track

## Κατάσταση: ΑΝΕΝΕΡΓΟ — δεν χρησιμοποιείται στο τρέχον pipeline

Αυτός ο φάκελος υπάρχει ως ΕΦΕΔΡΙΚΟ εργαλείο, όχι ως ενεργό κομμάτι του ICHNOS pipeline.

## Γιατί υπάρχει

Αρχικά εξετάστηκε η ιδέα να χρησιμοποιηθεί ένα Biologically-Informed Neural Network
(BINN, Lagergren et al. 2020, PLOS Comp Biol) ή ένα Universal Differential Equation
(UDE) για να "ανακαλύψει" τη μορφή ενός όρου adaptation/delay στο ER module, με βάση
την ομοιότητα του UPR adaptation με το delay term που βρήκε το BINN paper σε scratch
assay δεδομένα.

## Γιατί ΔΕΝ ενεργοποιήθηκε (μέχρι στιγμής)

1. Το πρόβλημα του BINN paper (χωρικό PDE, μετρούμενο u, delay όρος σε γνωστή θέση)
   δεν αντιστοιχεί ακριβώς στο δικό μας (well-mixed ODE, μη παρατηρούμενες καταστάσεις
   Ire1/Hac1/BiP, reporter πολλά βήματα κάτω από το sensing).
2. Το BINN paper's "ανακάλυψη" ήταν στην πραγματικότητα user-guided conjecture
   (πρόσθεσαν χειροκίνητα τον όρο T(t) μετά την παρατήρηση residuals), όχι αυτόματη.
3. Είχαμε ήδη μηχανιστικά υποψήφια μοντέλα (M0-M2, βλ. pulse-adaptation branch,
   fit_er_pincus_clearance.py) που εξηγούν το tail bias χωρίς νευρωνικό — clearance
   εισόδου με n σταθερό, ίδιος αριθμός παραμέτρων με το baseline.

## Πότε ΘΑ ενεργοποιηθεί αυτό το track

Μόνο αν το fit_er_pincus_clearance.py (pulse-adaptation branch) δείξει ότι ΚΑΝΕΝΑ
από τα M0/M1/M2 δεν εξηγεί επαρκώς το residual pattern — δηλαδή αν υπάρχει δομημένο
residual που καμία μηχανιστική υπόθεση δεν πιάνει. Σε αυτή την περίπτωση, ένα μικρό
UDE-style residual term πάνω στο ήδη υπάρχον ODE (όχι πλήρες BINN) θα ήταν το επόμενο
βήμα, με:
- Περιορισμούς πρόσημου/ορίων στην έξοδο του δικτύου (bio-informed constraints)
- Leave-one-dose-out validation (όχι τυχαίο split, με μόνο 2 δόσεις δεν έχει νόημα)
- Επιστροφή σε παραμετρική μορφή στο τέλος, όχι black-box παραμονή

## Αναφορές
- Lagergren, Nardini, Baker, Simpson & Flores (2020). "Biologically-informed neural
  networks guide mechanistic modeling from sparse experimental data." PLOS
  Computational Biology 16(12):e1008462.
