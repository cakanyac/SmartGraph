package fr.mael3il.api;

import fr.mael3il.objets.Observation;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.Response.ResponseBuilder;
import org.eclipse.microprofile.context.ThreadContext;
import org.neo4j.driver.Driver;
import org.neo4j.driver.Values;
import org.neo4j.driver.async.AsyncSession;

import java.util.List;
import java.util.concurrent.CompletionStage;

@Path("api/v1/observations")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class ApiObservations {

    @Inject
    Driver driver;

    @Inject
    ThreadContext threadContext;

    @GET
    public CompletionStage<Response> getAllObservations() {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<List<Observation>> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync("MATCH (o:Observation:SSN_Observation) RETURN o")
                        .thenCompose(cursor ->
                                cursor.listAsync(record ->
                                        Observation.from(record.get("o").asNode())
                                )));

        return threadContext.withContextCapture(cs)
                .thenCompose(obs ->
                        session.closeAsync().thenApply(signal -> obs))
                .thenApply(Response::ok)
                .thenApply(ResponseBuilder::build);
    }

    @GET
    @Path("/{id}")
    public CompletionStage<Response> getObservationById(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Observation> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync(
                                "MATCH (o:Observation:SSN_Observation {id: $id}) RETURN o",
                                Values.parameters("id", id)
                        )
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Observation.from(record.get("o").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(obs ->
                        session.closeAsync().thenApply(signal -> obs))
                .thenApply(obs -> {
                    if (obs == null) {
                        return Response.status(Response.Status.NOT_FOUND).build();
                    }
                    return Response.ok(obs).build();
                })
                .exceptionally(ex ->
                        Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    public CompletionStage<Response> createObservation(Observation observation) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Observation> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                CREATE (o:Observation:SSN_Observation {
                                  id: randomUUID(),
                                  resultTime: $resultTime,
                                  unit: $unit,
                                  observationType: $observationType,
                                  madeBySensor: $madeBySensor,
                                  numericValue: $numericValue
                                })
                                RETURN o
                                """,
                                Values.parameters(
                                        "resultTime", observation.getResultTime(),
                                        "unit", observation.getUnit(),
                                        "observationType", observation.getObservationType(),
                                        "madeBySensor", observation.getMadeBySensor(),
                                        "numericValue", observation.getNumericValue()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Observation.from(record.get("o").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(created ->
                        session.closeAsync().thenApply(signal -> created))
                .thenApply(created ->
                        Response.status(Response.Status.CREATED)
                                .entity(created)
                                .build());
    }

    @PUT
    @Path("/{id}")
    public CompletionStage<Response> updateObservation(@PathParam("id") String id, Observation observation) {
        AsyncSession session = driver.session(AsyncSession.class);
        CompletionStage<Observation> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                MATCH (o:Observation:SSN_Observation {id: $id})
                                SET o.resultTime = $resultTime,
                                    o.unit = $unit,
                                    o.observationType = $observationType,
                                    o.madeBySensor = $madeBySensor,
                                    o.numericValue = $numericValue
                                RETURN o
                                """,
                                Values.parameters(
                                        "id", id,
                                        "resultTime", observation.getResultTime(),
                                        "unit", observation.getUnit(),
                                        "observationType", observation.getObservationType(),
                                        "madeBySensor", observation.getMadeBySensor(),
                                        "numericValue", observation.getNumericValue()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Observation.from(record.get("o").asNode())
                                        )));
        return threadContext.withContextCapture(cs)
                .thenCompose(updatedObservation ->
                        session.closeAsync().thenApply(signal -> updatedObservation))
                .thenApply(updatedObservation ->
                        Response.ok(updatedObservation).build());
    }

    @DELETE
    @Path("/{id}")
    public CompletionStage<Response> deleteObservation(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Void> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                "MATCH (o:Observation:SSN_Observation {id: $id}) DELETE o",
                                Values.parameters("id", id)
                        )
                        .thenApply(r -> null));

        return threadContext.withContextCapture(cs)
                .thenCompose(signal ->
                        session.closeAsync().thenApply(s -> signal))
                .thenApply(signal ->
                        Response.ok().build());
    }
}
