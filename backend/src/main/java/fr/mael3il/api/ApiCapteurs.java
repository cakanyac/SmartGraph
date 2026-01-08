package fr.mael3il.api;

import fr.mael3il.objets.Capteur;
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

@Path("api/v1/capteurs")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class ApiCapteurs {

    @Inject
    Driver driver;

    @Inject
    ThreadContext threadContext;

    @GET
    public CompletionStage<Response> getAllCapteurs() {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<List<Capteur>> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync("MATCH (c:Capteur:SSN_Sensor) RETURN c")
                        .thenCompose(cursor ->
                                cursor.listAsync(record ->
                                        Capteur.from(record.get("c").asNode())
                                )));

        return threadContext.withContextCapture(cs)
                .thenCompose(capteurs ->
                        session.closeAsync().thenApply(signal -> capteurs))
                .thenApply(Response::ok)
                .thenApply(ResponseBuilder::build);
    }

    @GET
    @Path("/{id}")
    public CompletionStage<Response> getCapteurById(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Capteur> cs = session
                .executeReadAsync(tx -> tx
                        .runAsync(
                                "MATCH (c:Capteur:SSN_Sensor {id: $id}) RETURN c",
                                Values.parameters("id", id)
                        )
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Capteur.from(record.get("c").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(capteur ->
                        session.closeAsync().thenApply(signal -> capteur))
                .thenApply(capteur -> {
                    if (capteur == null) {
                        return Response.status(Response.Status.NOT_FOUND).build();
                    }
                    return Response.ok(capteur).build();
                })
                .exceptionally(ex ->
                        Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    public CompletionStage<Response> createCapteur(Capteur capteur) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Capteur> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                CREATE (c:Capteur:SSN_Sensor {
                                  id: randomUUID(),
                                  parcelle_id: $parcelle_id,
                                  type: $type,
                                  statut: $statut
                                })
                                RETURN c
                                """,
                                Values.parameters(
                                        "parcelle_id", capteur.getParcelleId(),
                                        "type", capteur.getType(),
                                        "statut", capteur.getStatut()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Capteur.from(record.get("c").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(created ->
                        session.closeAsync().thenApply(signal -> created))
                .thenApply(created ->
                        Response.status(Response.Status.CREATED).entity(created).build());
    }

    @PUT
    @Path("/{id}")
    public CompletionStage<Response> updateCapteur(@PathParam("id") String id, Capteur capteur) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Capteur> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                """
                                MATCH (c:Capteur:SSN_Sensor {id: $id})
                                SET c.type = $type,
                                    c.statut = $statut,
                                    c.parcelle_id = $parcelle_id
                                RETURN c
                                """,
                                Values.parameters(
                                        "id", id,
                                        "type", capteur.getType(),
                                        "statut", capteur.getStatut(),
                                        "parcelle_id", capteur.getParcelleId()
                                ))
                        .thenCompose(cursor ->
                                cursor.singleAsync()
                                        .thenApply(record ->
                                                Capteur.from(record.get("c").asNode())
                                        )));

        return threadContext.withContextCapture(cs)
                .thenCompose(updated ->
                        session.closeAsync().thenApply(signal -> updated))
                .thenApply(updated ->
                        Response.ok(updated).build());
    }

    @DELETE
    @Path("/{id}")
    public CompletionStage<Response> deleteCapteur(@PathParam("id") String id) {
        AsyncSession session = driver.session(AsyncSession.class);

        CompletionStage<Void> cs = session
                .executeWriteAsync(tx -> tx
                        .runAsync(
                                "MATCH (c:Capteur:SSN_Sensor {id: $id}) DELETE c",
                                Values.parameters("id", id)
                        )
                        .thenApply(result -> null));

        return threadContext.withContextCapture(cs)
                .thenCompose(signal ->
                        session.closeAsync().thenApply(s -> signal))
                .thenApply(signal ->
                        Response.ok().build());
    }
}
